class VertexTF:
    def __init__(self, project_id, epochs):
        self.project_id = project_id
        self.epochs = epochs
    
    def query(self, start_date, end_date):
        import os
        import numpy as np
        from google.cloud import bigquery

        query = f"""
            WITH all_visitor_stats AS (
                SELECT
                  fullvisitorid,
                  IF(COUNTIF(totals.transactions > 0 AND totals.newVisits IS NULL) > 0, 1, 0) AS will_buy_on_return_visit
                FROM `data-to-insights.ecommerce.web_analytics`
                GROUP BY fullvisitorid
            )
            # add in new features
            SELECT 
                * EXCEPT(unique_session_id) 
            FROM (
                SELECT
                    CONCAT(fullvisitorid, CAST(visitId AS STRING)) AS unique_session_id,
                    # labels
                    will_buy_on_return_visit,
                    MAX(CAST(h.eCommerceAction.action_type AS INT64)) AS latest_ecommerce_progress,
                    # behavior on the site
                    IFNULL(totals.bounces, 0) AS bounces,
                    IFNULL(totals.timeOnSite, 0) AS time_on_site,
                    IFNULL(totals.pageviews, 0) AS pageviews,
                    # where the visitor came from
                    trafficSource.source,
                    trafficSource.medium,
                    channelGrouping,
                    # mobile or desktop
                    device.deviceCategory,
                    # geographic
                    IFNULL(geoNetwork.country, "") AS country
                FROM `data-to-insights.ecommerce.web_analytics`,
                    UNNEST(hits) AS h
                    JOIN all_visitor_stats USING(fullvisitorid)
                WHERE 1=1
                    # only predict for new visits
                    AND totals.newVisits = 1
                    AND date BETWEEN '{start_date}' AND '{end_date}' # train 9 months
                GROUP BY
                    unique_session_id,
                    will_buy_on_return_visit,
                    bounces,
                    time_on_site,
                    totals.pageviews,
                    trafficSource.source,
                    trafficSource.medium,
                    channelGrouping,
                    device.deviceCategory,
                    country)
        """
        
        '''Split and Transform data into tf.Dataset, shuffles + batch'''
    
        if os.getenv('CLOUD_ML_PROJECT') is not None:
            project = os.environ['CLOUD_ML_PROJECT']
        else: project = self.project_id
        client = bigquery.Client(project=project)
        self.df = client.query(query).to_dataframe()
        self.train, self.val, self.test = np.split(self.df.sample(frac=1), [int(0.8*len(self.df)), int(0.9*len(self.df))])
        
        
        return self.train, self.val, self.test

    def preprocessing(self, target_column, batch_size=32, shuffle=True):
        import numpy as np
        import tensorflow as tf
        from src.data.Transform import df_to_dataset, get_category_encoding_layer, get_normalization_layer
        
        self.train_ds = df_to_dataset(self.train, batch_size=batch_size, target_column=target_column, shuffle=True)
        self.val_ds = df_to_dataset(self.val, batch_size=batch_size, target_column=target_column, shuffle=True)
        self.test_ds = df_to_dataset(self.test, batch_size=batch_size, target_column=target_column, shuffle=True)
        
        self.all_inputs = []
        self.encoded_features = []

        # Numerical features.
        cat_columns = [i for i in self.df if self.df[i].dtypes == 'object' and i != target_column]
        num_columns = [i for i in self.df if self.df[i].dtypes == 'int64' and i != target_column]

        print(cat_columns)
        print(num_columns)

        for header in num_columns:
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = get_normalization_layer(header, self.train_ds)
            encoded_numeric_col = normalization_layer(numeric_col)
            self.all_inputs.append(numeric_col)
            self.encoded_features.append(encoded_numeric_col)

        for header in cat_columns:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
            encoding_layer = get_category_encoding_layer(
            name=header,
            dataset=self.train_ds,
            dtype='string',
            max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        self.all_inputs.append(categorical_col)
        self.encoded_features.append(encoded_categorical_col)

        return self.train_ds, self.val_ds, self.test_ds 

    def create_model(self, nn_input: int, lr: float):
        import tensorflow as tf

        '''Train model with TF+Keras'''
        all_features = tf.keras.layers.concatenate(self.encoded_features)
        x = tf.keras.layers.Dense(nn_input, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(self.all_inputs, output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"])
        return model