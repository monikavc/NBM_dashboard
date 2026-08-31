## in this script, PRISM 4km resolution files containing shapefiles are fetched and unzipped for the past 60 days. pushed to my s3 bucket at
##    https://us-east-1.console.aws.amazon.com/s3/buckets/prism-dataset-4km?region=us-east-1&tab=objects

from datetime import datetime, timedelta
import os
import io
import zipfile
import requests
import boto3

BUCKET_NAME = "prism-dataset-4km"
START_DATE = datetime.date.today() - datetime.timedelta(days=61)
END_DATE = datetime.date.today()
RESOLUTION = "4km"
VARIABLES = ["tmax"]

s3_client = boto3.client('s3')

def download_and_upload_prism():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    current_date = start
    base_url = f"https://services.nacse.org/prism/data/get/us/{RESOLUTION}"

    while current_date <= end:
        date_str = current_date.strftime("%Y%m%d")
        for var in VARIABLES:
            url = f"{base_url}/{var}/{date_str}"
            s3_key_prefix = f"{current_date.year}/{var}/"
     
            print(f"retreiving {var} for {current_date.strftime('%Y-%m-%d')}")

            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    print(f"  Skipping {date_str}: PRISM API returned status {response.status_code}")
                    continue
                
                zip_file = zipfile.ZipFile(io.BytesIO(response.content))
                
                for file_name in zip_file.namelist():
                    s3_destination = os.path.join(s3_key_prefix, file_name)
                    file_content = zip_file.read(file_name)
                    
                    s3_client.put_object(
                        Bucket=BUCKET_NAME,
                        Key=s3_destination,
                        Body=file_content
                    )
                print(f"Successfully pushed {date_str} files to S3 bucket!")
                
            except Exception as e:
                print(f"Error on {date_str}: {e}")
                
        current_date += timedelta(days=1)

if __name__ == "__main__":
    print("Starting data bucket input run")
    download_and_upload_prism()
    print("Run complet")