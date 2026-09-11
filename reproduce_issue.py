
from unittest import mock
import time
from functools import wraps

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("=== Entering with_retry wrapper ===")
            attempt = 0
            while attempt < max_attempts:
                print(f"Attempt {attempt+1}/{max_attempts}")
                try:
                    print("Calling func...")
                    result = func(*args, **kwargs)
                    print(f"func returned: {result}")
                    return result
                except exceptions as e:
                    print(f"Caught exception in with_retry: {e}, type: {type(e)}")
                    attempt += 1
                    if attempt >= max_attempts:
                        print(f"Max attempts ({max_attempts}) reached for {func.__name__}")
                        raise
                    
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    print(f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. Retrying...")
                    time.sleep(delay)
            print("Exiting with_retry wrapper with None")
            return None
        return wrapper
    return decorator

class S3Storage:
    @with_retry(max_attempts=2)
    def file_exists(self, remote_path: str, bucket: str) -> bool:
        print("=== Entering S3Storage.file_exists ===")
        if not self._client:
            return False
        
        try:
            print("Calling self._client.head_object...")
            self._client.head_object(Bucket=bucket, Key=remote_path)
            print("head_object didn't raise, returning True")
            return True
        except Exception as e:
            print("Caught exception in file_exists!")
            print(f"Exception type: {type(e)}")
            print(f"Exception type name: {type(e).__name__}")
            print(f"self._client.exceptions: {self._client.exceptions}")
            print(f"self._client.exceptions.NoSuchKey: {self._client.exceptions.NoSuchKey}")
            if hasattr(self._client, 'exceptions') and hasattr(self._client.exceptions, 'NoSuchKey'):
                try:
                    print(f"Checking isinstance(e, self._client.exceptions.NoSuchKey): {isinstance(e, self._client.exceptions.NoSuchKey)}")
                    if isinstance(e, self._client.exceptions.NoSuchKey):
                        print("Returning False because it's NoSuchKey!")
                        return False
                except Exception as check_error:
                    print(f"Exception when checking isinstance: {check_error}")
                    if type(e).__name__ == 'NoSuchKey':
                        print("Returning False because exception name is NoSuchKey!")
                        return False
            print(f"Returning False for error: {e}")
            return False

# Now test it
if __name__ == "__main__":
    storage = S3Storage()
    storage._client = mock.MagicMock()
    storage._client.exceptions = mock.MagicMock()
    storage._client.head_object.side_effect = storage._client.exceptions.NoSuchKey()

    print("Calling file_exists...")
    result = storage.file_exists("test/file.txt", "test-bucket")
    print(f"Result: {result}")

