import sys
from google.cloud import storage
import os
import json
import logging



class GSCBucketConnectorClass():

    def __init__(self):
        try:
            Path_to_credential = "path_to_the_cloud_service_account_credential"
            self.storage_client = storage.Client.from_service_account_json(Path_to_credential)  
        except:
            print(f"Failed to Initiate the GSCBucketConnectorClass ")


    def download_blob_4_detection(self,bucket_name,source_blob_Folder_name,destination_file_name):
        bucket=self.storage_client.get_bucket(bucket_name)
        blobs=list(bucket.list_blobs(prefix=source_blob_Folder_name))
        try:
            for blob in blobs:
                if(not blob.name.endswith("/")):
                    blob.download_to_filename(destination_file_name+blob.name)
        except:
            print(f"Failed to download the from bucket ")

bucket_name = "image_input"
destination_file_name = "path_to/dataset_classifer/croppedface_14-12-2020/" 

source_blob_Folder_name = "path_to_GCS_source_bucket/2020-12-08/"
# Create this folder locally
if not os.path.exists(destination_file_name+source_blob_Folder_name):
    os.makedirs(destination_file_name+source_blob_Folder_name)

gcs=GSCBucketConnectorClass()
gcs.download_blob_4_detection(bucket_name,source_blob_Folder_name,destination_file_name)
            


        
