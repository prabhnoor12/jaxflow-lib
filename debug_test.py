
from unittest import mock
from jaxflow.advanced.cloud import S3Storage, CloudCredentials

def test_debug():
    print("Starting test...")
    cloud_creds = CloudCredentials(
        provider="s3",
        access_key="test-access-key",
        secret_key="test-secret-key",
        region="us-east-1"
    )
    with mock.patch('jaxflow.advanced.cloud.boto3'):
        storage = S3Storage(cloud_creds)
        storage._client = mock.MagicMock()
        storage._client.exceptions = mock.MagicMock()
        storage._client.head_object.side_effect = storage._client.exceptions.NoSuchKey()
        print("Calling file_exists...")
        result = storage.file_exists("test/file.txt", "test-bucket")
        print(f"Result is {result}")
        assert result is False

if __name__ == "__main__":
    test_debug()
