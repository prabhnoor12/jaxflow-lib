"""
Comprehensive tests for the cloud integration suite.

This test suite includes:
- Unit tests with mocks (no real cloud credentials needed)
- Optional integration tests (requires real credentials)
- Tests for all three cloud providers (S3, GCS, Azure)
"""
import datetime
import os
import sys
import tempfile
import hashlib
import pytest
from unittest import mock
from pathlib import Path
from datetime import datetime

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxflow.advanced.cloud import (
    CloudCredentials,
    FileMetadata,
    CloudManager,
    S3Storage,
    GCSStorage,
    AzureBlobStorage,
    with_retry,
)
from jaxflow.advanced.licensing import LicenseManager


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def temp_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
        f.write(b'Test file content for cloud operations')
    path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_metadata():
    """Create sample file metadata."""
    return FileMetadata(
        path="test/file.txt",
        size=1024,
        checksum="d41d8cd98f00b204e9800998ecf8427e",
        last_modified=datetime.now().isoformat(),
        content_type="text/plain"
    )


@pytest.fixture
def mock_license_manager():
    """Create a mock license manager that allows all features."""
    lm = mock.MagicMock(spec=LicenseManager)
    lm.has_feature.return_value = True
    return lm


@pytest.fixture
def cloud_credentials_s3():
    """Create sample S3 cloud credentials."""
    return CloudCredentials(
        provider="s3",
        access_key="test-access-key",
        secret_key="test-secret-key",
        region="us-east-1"
    )


@pytest.fixture
def cloud_credentials_gcs():
    """Create sample GCS cloud credentials."""
    return CloudCredentials(
        provider="gcs",
        service_account_json="/path/to/creds.json"
    )


@pytest.fixture
def cloud_credentials_azure():
    """Create sample Azure cloud credentials."""
    return CloudCredentials(provider="azure")


# =============================================================================
# UNIT TESTS - CORE CLASSES
# =============================================================================

class TestCloudCredentials:
    """Tests for the CloudCredentials class."""

    def test_initialization(self):
        """Test basic initialization."""
        creds = CloudCredentials(
            provider="s3",
            access_key="test-access",
            secret_key="test-secret",
            region="us-east-1"
        )
        assert creds.provider == "s3"
        assert creds.access_key == "test-access"
        assert creds.secret_key == "test-secret"
        assert creds.region == "us-east-1"

    @mock.patch.dict(os.environ, {
        "AWS_ACCESS_KEY_ID": "env-access",
        "AWS_SECRET_ACCESS_KEY": "env-secret",
        "AWS_REGION": "env-region"
    })
    def test_from_env_s3(self):
        """Test loading S3 credentials from environment."""
        creds = CloudCredentials.from_env("s3")
        assert creds.provider == "s3"
        assert creds.access_key == "env-access"
        assert creds.secret_key == "env-secret"
        assert creds.region == "env-region"

    @mock.patch.dict(os.environ, {
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"
    })
    def test_from_env_gcs(self):
        """Test loading GCS credentials from environment."""
        creds = CloudCredentials.from_env("gcs")
        assert creds.provider == "gcs"
        assert creds.service_account_json == "/path/to/creds.json"

    def test_from_env_invalid_provider(self):
        """Test invalid provider raises error."""
        with pytest.raises(ValueError):
            CloudCredentials.from_env("invalid-provider")


class TestFileMetadata:
    """Tests for the FileMetadata class."""

    def test_initialization(self, sample_metadata):
        """Test basic initialization."""
        assert sample_metadata.path == "test/file.txt"
        assert sample_metadata.size == 1024
        assert sample_metadata.checksum is not None
        assert sample_metadata.last_modified is not None


class TestRetryDecorator:
    """Tests for the with_retry decorator."""

    def test_successful_operation(self):
        """Test decorator works with successful function."""
        call_count = [0]

        @with_retry(max_attempts=3)
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count[0] == 1

    def test_eventual_success(self):
        """Test function that succeeds after retries."""
        call_count = [0]

        @with_retry(max_attempts=3, base_delay=0.01)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count[0] == 3

    def test_always_fails(self):
        """Test function that always fails raises after max attempts."""
        call_count = [0]

        @with_retry(max_attempts=3, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            raise Exception("Always fails")

        with pytest.raises(Exception, match="Always fails"):
            failing_func()
        assert call_count[0] == 3


# =============================================================================
# MOCK TESTS - S3 STORAGE
# =============================================================================

class TestS3Storage:
    """Tests for S3Storage using mocks."""

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_initialization(self, mock_boto3, cloud_credentials_s3):
        """Test S3Storage initialization."""
        storage = S3Storage(cloud_credentials_s3)
        assert storage is not None
        mock_boto3.client.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_file_exists_true(self, mock_boto3, cloud_credentials_s3):
        """Test file_exists returns True when file exists."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()
        storage._client.head_object.return_value = True

        exists = storage.file_exists("test/file.txt", "test-bucket")
        assert exists is True
        storage._client.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/file.txt"
        )

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_file_exists_false(self, mock_boto3, cloud_credentials_s3):
        """Test file_exists returns False when file does not exist."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()
        storage._client.exceptions = mock.MagicMock()
        storage._client.head_object.side_effect = storage._client.exceptions.NoSuchKey()

        exists = storage.file_exists("test/file.txt", "test-bucket")
        assert exists is False

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_upload_file(self, mock_boto3, cloud_credentials_s3, temp_file):
        """Test file upload."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()
        storage._transfer_config = mock.MagicMock()

        success = storage.upload_file(
            str(temp_file),
            "test/file.txt",
            "test-bucket",
            verify_checksum=False
        )
        assert success is True
        storage._client.upload_file.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_download_file(self, mock_boto3, cloud_credentials_s3, temp_file, temp_dir):
        """Test file download."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()
        storage._transfer_config = mock.MagicMock()
        
        # Mock get_metadata
        metadata = FileMetadata(
            path="test/file.txt",
            size=temp_file.stat().st_size,
            checksum=S3Storage.compute_checksum(str(temp_file))
        )
        storage.get_metadata = mock.MagicMock(return_value=metadata)
        
        local_path = str(temp_dir / "downloaded.txt")
        success = storage.download_file(
            "test/file.txt",
            local_path,
            "test-bucket",
            verify_checksum=False
        )
        assert success is True
        storage._client.download_file.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_list_files(self, mock_boto3, cloud_credentials_s3):
        """Test listing files."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()
        storage._client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'file1.txt', 'Size': 100, 'LastModified': datetime.now()},
                {'Key': 'file2.txt', 'Size': 200, 'LastModified': datetime.now()},
            ],
            'IsTruncated': False
        }

        files = storage.list_files("", "test-bucket")
        assert len(files) == 2
        assert files[0].path == "file1.txt"
        assert files[1].path == "file2.txt"

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_delete_file(self, mock_boto3, cloud_credentials_s3):
        """Test file deletion."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()

        success = storage.delete_file("test/file.txt", "test-bucket")
        assert success is True
        storage._client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test/file.txt"
        )

    @mock.patch('jaxflow.advanced.cloud.boto3')
    def test_get_metadata(self, mock_boto3, cloud_credentials_s3):
        """Test getting file metadata."""
        storage = S3Storage(cloud_credentials_s3)
        storage._client = mock.MagicMock()
        storage._client.head_object.return_value = {
            'ContentLength': 100,
            'LastModified': datetime.now(),
            'Metadata': {'checksum-md5': 'test-checksum'},
            'ContentType': 'text/plain'
        }

        metadata = storage.get_metadata("test/file.txt", "test-bucket")
        assert metadata is not None
        assert metadata.size == 100
        assert metadata.checksum == 'test-checksum'


# =============================================================================
# MOCK TESTS - GCS STORAGE
# =============================================================================

class TestGCSStorage:
    """Tests for GCSStorage using mocks."""

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_initialization(self, mock_storage, cloud_credentials_gcs):
        """Test GCSStorage initialization."""
        storage = GCSStorage(cloud_credentials_gcs)
        assert storage is not None
        mock_storage.Client.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_file_exists_true(self, mock_storage, cloud_credentials_gcs):
        """Test file_exists returns True when file exists."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        storage._client.bucket.return_value = mock_bucket

        exists = storage.file_exists("test/file.txt", "test-bucket")
        assert exists is True

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_file_exists_false(self, mock_storage, cloud_credentials_gcs):
        """Test file_exists returns False when file does not exist."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        storage._client.bucket.return_value = mock_bucket

        exists = storage.file_exists("test/file.txt", "test-bucket")
        assert exists is False

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_upload_file(self, mock_storage, cloud_credentials_gcs, temp_file):
        """Test file upload."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_bucket.blob.return_value = mock_blob
        storage._client.bucket.return_value = mock_bucket

        success = storage.upload_file(
            str(temp_file),
            "test/file.txt",
            "test-bucket",
            verify_checksum=False
        )
        assert success is True
        mock_blob.upload_from_filename.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_download_file(self, mock_storage, cloud_credentials_gcs, temp_file, temp_dir):
        """Test file download."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.size = temp_file.stat().st_size
        mock_blob.metadata = {'checksum-md5': GCSStorage.compute_checksum(str(temp_file))}
        mock_bucket.blob.return_value = mock_blob
        storage._client.bucket.return_value = mock_bucket

        local_path = str(temp_dir / "downloaded.txt")
        success = storage.download_file(
            "test/file.txt",
            local_path,
            "test-bucket",
            verify_checksum=False
        )
        assert success is True
        mock_blob.download_to_filename.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_list_files(self, mock_storage, cloud_credentials_gcs):
        """Test listing files."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        
        mock_blob1 = mock.MagicMock()
        mock_blob1.name = "file1.txt"
        mock_blob1.size = 100
        mock_blob1.updated = datetime.now()
        mock_blob1.metadata = None
        mock_blob1.content_type = "text/plain"
        
        mock_blob2 = mock.MagicMock()
        mock_blob2.name = "file2.txt"
        mock_blob2.size = 200
        mock_blob2.updated = datetime.now()
        mock_blob2.metadata = None
        mock_blob2.content_type = "text/plain"
        
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        storage._client.bucket.return_value = mock_bucket

        files = storage.list_files("", "test-bucket")
        assert len(files) == 2
        assert files[0].path == "file1.txt"
        assert files[1].path == "file2.txt"

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_delete_file(self, mock_storage, cloud_credentials_gcs):
        """Test file deletion."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_bucket.blob.return_value = mock_blob
        storage._client.bucket.return_value = mock_bucket

        success = storage.delete_file("test/file.txt", "test-bucket")
        assert success is True
        mock_blob.delete.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.storage')
    def test_get_metadata(self, mock_storage, cloud_credentials_gcs):
        """Test getting file metadata."""
        storage = GCSStorage(cloud_credentials_gcs)
        storage._client = mock.MagicMock()
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()
        mock_blob.size = 100
        mock_blob.updated = datetime.now()
        mock_blob.metadata = {'checksum-md5': 'test-checksum'}
        mock_blob.content_type = "text/plain"
        mock_bucket.get_blob.return_value = mock_blob
        storage._client.bucket.return_value = mock_bucket

        metadata = storage.get_metadata("test/file.txt", "test-bucket")
        assert metadata is not None
        assert metadata.size == 100
        assert metadata.checksum == 'test-checksum'


# =============================================================================
# MOCK TESTS - AZURE BLOB STORAGE
# =============================================================================

class TestAzureBlobStorage:
    """Tests for AzureBlobStorage using mocks."""

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_initialization(self, mock_blob_service, cloud_credentials_azure):
        """Test AzureBlobStorage initialization."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        assert storage is not None
        mock_blob_service.from_connection_string.assert_called_once()

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_file_exists_true(self, mock_blob_service, cloud_credentials_azure):
        """Test file_exists returns True when file exists."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_blob_client = mock.MagicMock()
        mock_blob_client.exists.return_value = True
        storage._service_client.get_blob_client.return_value = mock_blob_client

        exists = storage.file_exists("test/file.txt", "test-container")
        assert exists is True

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_file_exists_false(self, mock_blob_service, cloud_credentials_azure):
        """Test file_exists returns False when file does not exist."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_blob_client = mock.MagicMock()
        mock_blob_client.exists.return_value = False
        storage._service_client.get_blob_client.return_value = mock_blob_client

        exists = storage.file_exists("test/file.txt", "test-container")
        assert exists is False

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_upload_file(self, mock_blob_service, cloud_credentials_azure, temp_file):
        """Test file upload."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_blob_client = mock.MagicMock()
        storage._service_client.get_blob_client.return_value = mock_blob_client

        success = storage.upload_file(
            str(temp_file),
            "test/file.txt",
            "test-container",
            verify_checksum=False
        )
        assert success is True
        mock_blob_client.upload_blob.assert_called_once()

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_download_file(self, mock_blob_service, cloud_credentials_azure, temp_file, temp_dir):
        """Test file download."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_blob_client = mock.MagicMock()
        mock_blob_client.exists.return_value = True
        
        mock_properties = mock.MagicMock()
        mock_properties.size = temp_file.stat().st_size
        mock_properties.metadata = {'checksum-md5': AzureBlobStorage.compute_checksum(str(temp_file))}
        mock_blob_client.get_blob_properties.return_value = mock_properties
        
        mock_downloader = mock.MagicMock()
        mock_blob_client.download_blob.return_value = mock_downloader
        
        storage._service_client.get_blob_client.return_value = mock_blob_client

        local_path = str(temp_dir / "downloaded.txt")
        success = storage.download_file(
            "test/file.txt",
            local_path,
            "test-container",
            verify_checksum=False
        )
        assert success is True
        mock_blob_client.download_blob.assert_called_once()

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_list_files(self, mock_blob_service, cloud_credentials_azure):
        """Test listing files."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_container_client = mock.MagicMock()
        
        mock_blob1 = mock.MagicMock()
        mock_blob1.name = "file1.txt"
        mock_blob1.size = 100
        mock_blob1.last_modified = datetime.now()
        mock_blob1.metadata = None
        mock_blob1.content_settings = mock.MagicMock()
        mock_blob1.content_settings.content_type = "text/plain"
        
        mock_blob2 = mock.MagicMock()
        mock_blob2.name = "file2.txt"
        mock_blob2.size = 200
        mock_blob2.last_modified = datetime.now()
        mock_blob2.metadata = None
        mock_blob2.content_settings = mock.MagicMock()
        mock_blob2.content_settings.content_type = "text/plain"
        
        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]
        storage._service_client.get_container_client.return_value = mock_container_client

        files = storage.list_files("", "test-container")
        assert len(files) == 2
        assert files[0].path == "file1.txt"
        assert files[1].path == "file2.txt"

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_delete_file(self, mock_blob_service, cloud_credentials_azure):
        """Test file deletion."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_blob_client = mock.MagicMock()
        storage._service_client.get_blob_client.return_value = mock_blob_client

        success = storage.delete_file("test/file.txt", "test-container")
        assert success is True
        mock_blob_client.delete_blob.assert_called_once()

    @mock.patch.dict(os.environ, {
        "AZURE_STORAGE_CONNECTION_STRING": "test-connection-string"
    })
    @mock.patch('jaxflow.advanced.cloud.BlobServiceClient')
    def test_get_metadata(self, mock_blob_service, cloud_credentials_azure):
        """Test getting file metadata."""
        storage = AzureBlobStorage(cloud_credentials_azure)
        storage._service_client = mock.MagicMock()
        mock_blob_client = mock.MagicMock()
        
        mock_properties = mock.MagicMock()
        mock_properties.size = 100
        mock_properties.last_modified = datetime.now()
        mock_properties.metadata = {'checksum-md5': 'test-checksum'}
        mock_properties.content_settings = mock.MagicMock()
        mock_properties.content_settings.content_type = "text/plain"
        
        mock_blob_client.get_blob_properties.return_value = mock_properties
        storage._service_client.get_blob_client.return_value = mock_blob_client

        metadata = storage.get_metadata("test/file.txt", "test-container")
        assert metadata is not None
        assert metadata.size == 100
        assert metadata.checksum == 'test-checksum'


# =============================================================================
# MOCK TESTS - CLOUD MANAGER
# =============================================================================

class TestCloudManager:
    """Tests for CloudManager using mocks."""

    def test_initialization(self, mock_license_manager):
        """Test CloudManager initialization."""
        manager = CloudManager(
            license_manager=mock_license_manager,
            default_provider="s3"
        )
        assert manager is not None
        assert manager.default_provider == "s3"

    @mock.patch('jaxflow.advanced.cloud.S3Storage')
    def test_register_credentials(self, mock_s3_storage, cloud_credentials_s3, mock_license_manager):
        """Test registering credentials."""
        manager = CloudManager(license_manager=mock_license_manager)
        manager.register_credentials("s3", cloud_credentials_s3)
        assert "s3" in manager._storage_clients
        mock_s3_storage.assert_called_once_with(cloud_credentials_s3)

    @mock.patch.object(CloudManager, 'register_credentials')
    def test_register_credentials_from_env(self, mock_register, mock_license_manager):
        """Test registering credentials from environment."""
        manager = CloudManager(license_manager=mock_license_manager)
        with mock.patch.object(CloudCredentials, 'from_env') as mock_from_env:
            mock_creds = mock.MagicMock()
            mock_from_env.return_value = mock_creds
            manager.register_credentials_from_env("s3")
            mock_from_env.assert_called_once_with("s3")
            mock_register.assert_called_once_with("s3", mock_creds)

    def test_get_storage_unregistered(self, mock_license_manager):
        """Test get_storage raises error for unregistered provider."""
        manager = CloudManager(license_manager=mock_license_manager)
        with pytest.raises(ValueError, match="No credentials registered"):
            manager.get_storage("s3")

    @mock.patch('jaxflow.advanced.cloud.S3Storage')
    def test_sync_checkpoint_upload(self, mock_s3_storage, cloud_credentials_s3, mock_license_manager, temp_file):
        """Test sync_checkpoint upload."""
        manager = CloudManager(license_manager=mock_license_manager)
        mock_storage = mock.MagicMock()
        mock_storage.upload_file.return_value = True
        manager._storage_clients["s3"] = mock_storage
        
        success = manager.sync_checkpoint(
            str(temp_file),
            "checkpoint.pt",
            "test-bucket",
            provider="s3",
            upload=True
        )
        assert success is True
        mock_storage.upload_file.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.S3Storage')
    def test_sync_checkpoint_download(self, mock_s3_storage, cloud_credentials_s3, mock_license_manager, temp_dir):
        """Test sync_checkpoint download."""
        manager = CloudManager(license_manager=mock_license_manager)
        mock_storage = mock.MagicMock()
        mock_storage.download_file.return_value = True
        manager._storage_clients["s3"] = mock_storage
        
        local_path = str(temp_dir / "checkpoint.pt")
        success = manager.sync_checkpoint(
            local_path,
            "checkpoint.pt",
            "test-bucket",
            provider="s3",
            upload=False
        )
        assert success is True
        mock_storage.download_file.assert_called_once()

    @mock.patch('jaxflow.advanced.cloud.S3Storage')
    def test_stream_dataset_from_cloud(self, mock_s3_storage, cloud_credentials_s3, mock_license_manager, temp_dir):
        """Test streaming dataset from cloud with LRU cache."""
        manager = CloudManager(license_manager=mock_license_manager)
        mock_storage = mock.MagicMock()
        
        mock_storage.list_files.return_value = [
            FileMetadata(path="file1.txt", size=100, checksum="123"),
            FileMetadata(path="file2.txt", size=200, checksum="456"),
        ]
        mock_storage.download_file.return_value = True
        manager._storage_clients["s3"] = mock_storage

        dataset = manager.stream_dataset_from_cloud(
            remote_prefix="",
            local_cache_dir=str(temp_dir),
            bucket="test-bucket",
            provider="s3"
        )

        assert len(dataset) == 2
        assert hasattr(dataset, '__getitem__')
        assert hasattr(dataset, 'get_file_metadata')

    @mock.patch('jaxflow.advanced.cloud.S3Storage')
    def test_batch_download(self, mock_s3_storage, cloud_credentials_s3, mock_license_manager, temp_dir):
        """Test batch download."""
        manager = CloudManager(license_manager=mock_license_manager)
        mock_storage = mock.MagicMock()
        mock_storage.download_file.return_value = True
        manager._storage_clients["s3"] = mock_storage
        
        results = manager.batch_download(
            ["file1.txt", "file2.txt"],
            str(temp_dir),
            "test-bucket",
            provider="s3"
        )
        assert len(results) == 2
        assert all(results.values())

    @mock.patch('jaxflow.advanced.cloud.S3Storage')
    def test_batch_upload(self, mock_s3_storage, cloud_credentials_s3, mock_license_manager, temp_file):
        """Test batch upload."""
        manager = CloudManager(license_manager=mock_license_manager)
        mock_storage = mock.MagicMock()
        mock_storage.upload_file.return_value = True
        manager._storage_clients["s3"] = mock_storage
        
        results = manager.batch_upload(
            [str(temp_file)],
            "",
            "test-bucket",
            provider="s3"
        )
        assert len(results) == 1
        assert all(results.values())

    def test_get_cost_report(self, mock_license_manager):
        """Test getting cost report."""
        manager = CloudManager(license_manager=mock_license_manager)
        report = manager.get_cost_report()
        assert isinstance(report, dict)

    def test_get_operation_history(self, mock_license_manager):
        """Test getting operation history."""
        manager = CloudManager(license_manager=mock_license_manager)
        history = manager.get_operation_history()
        assert isinstance(history, list)


# =============================================================================
# UNIT TESTS - CHECKSUM VERIFICATION
# =============================================================================

class TestChecksumVerification:
    """Tests for checksum verification."""

    def test_compute_checksum(self, temp_file):
        """Test checksum computation."""
        checksum = S3Storage.compute_checksum(str(temp_file))
        assert isinstance(checksum, str)
        assert len(checksum) == 32  # MD5 hex digest length

    def test_checksum_consistency(self, temp_file):
        """Test same file produces same checksum."""
        checksum1 = S3Storage.compute_checksum(str(temp_file))
        checksum2 = S3Storage.compute_checksum(str(temp_file))
        assert checksum1 == checksum2

    def test_checksum_different_files(self, temp_file):
        """Test different files produce different checksums."""
        checksum1 = S3Storage.compute_checksum(str(temp_file))

        # Create different file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'Different content')
        temp_file2 = Path(f.name)

        checksum2 = S3Storage.compute_checksum(str(temp_file2))
        assert checksum1 != checksum2

        # Clean up
        temp_file2.unlink()


# =============================================================================
# INTEGRATION TESTS (OPTIONAL - REQUIRE REAL CREDENTIALS)
# =============================================================================

def _has_aws_credentials():
    """Check if AWS credentials are available."""
    return all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("AWS_S3_TEST_BUCKET")
    ])


@pytest.mark.skipif(not _has_aws_credentials(), reason="AWS credentials not available")
class TestS3Integration:
    """Integration tests for S3 storage (requires real credentials)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test with real credentials."""
        self.bucket = os.getenv("AWS_S3_TEST_BUCKET")
        self.test_prefix = f"jaxflow-test-{os.urandom(4).hex()}/"

        creds = CloudCredentials.from_env("s3")
        self.storage = S3Storage(creds)

    def test_full_workflow(self, temp_file):
        """Test complete upload-download-verify workflow."""
        remote_path = self.test_prefix + "test-file.txt"

        # Upload
        upload_success = self.storage.upload_file(
            str(temp_file),
            remote_path,
            self.bucket,
            verify_checksum=True
        )
        assert upload_success is True

        # Check exists
        exists = self.storage.file_exists(remote_path, self.bucket)
        assert exists is True

        # Get metadata
        metadata = self.storage.get_metadata(remote_path, self.bucket)
        assert metadata is not None
        assert metadata.path == remote_path
        assert metadata.size == temp_file.stat().st_size

        # Download
        with tempfile.NamedTemporaryFile(delete=False) as f:
            download_path = Path(f.name)
        download_success = self.storage.download_file(
            remote_path,
            str(download_path),
            self.bucket,
            verify_checksum=True
        )
        assert download_success is True

        # Verify content
        with open(temp_file, 'rb') as f1, open(download_path, 'rb') as f2:
            assert f1.read() == f2.read()

        # List files
        files = self.storage.list_files(self.test_prefix, self.bucket)
        assert len(files) >= 1
        assert any(f.path == remote_path for f in files)

        # Delete
        delete_success = self.storage.delete_file(remote_path, self.bucket)
        assert delete_success is True

        # Verify deletion
        exists_after = self.storage.file_exists(remote_path, self.bucket)
        assert exists_after is False

        # Clean up
        download_path.unlink()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
    ])
