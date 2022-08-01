from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
credentials = GoogleCredentials.get_application_default()
service = discovery.build('compute', 'v1', credentials=credentials)
request = service.regionCommitments().aggregatedList(project='jchavezar-demo')
request.execute()
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
credentials = GoogleCredentials.get_application_default()
service = discovery.build('compute', 'v1', credentials=credentials)
request = service.regionCommitments().aggregatedList(project='jchavezar-demo')
print(request.execute())
