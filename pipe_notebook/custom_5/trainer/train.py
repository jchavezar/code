
import os

print(os.listdir())
print(os.listdir('/mnt'))

with open('/mnt/nfs/nfs_models/1.txt', 'w') as f:
    f.write('test')
