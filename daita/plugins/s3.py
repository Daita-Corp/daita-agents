"""
AWS S3 plugin for Daita Agents.

Simple S3 object storage operations - no over-engineering.
"""
import logging
import os
import io
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)

class S3Plugin:
    """
    Simple AWS S3 plugin for agents.
    
    Handles S3 operations with automatic format detection and focus system support.
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize S3 connection.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            aws_access_key_id: AWS access key (optional, uses env/IAM if not provided)
            aws_secret_access_key: AWS secret key (optional, uses env/IAM if not provided)
            aws_session_token: AWS session token (optional, for temporary credentials)
            endpoint_url: Custom S3 endpoint URL (for S3-compatible services)
            **kwargs: Additional boto3 parameters
        """
        if not bucket or not bucket.strip():
            raise ValueError("S3 bucket name cannot be empty")
        
        self.bucket = bucket
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.endpoint_url = endpoint_url
        
        # Store additional config
        self.config = kwargs
        
        self._client = None
        self._session = None
        
        logger.debug(f"S3 plugin configured for bucket {bucket} in region {region}")
    
    async def connect(self):
        """Initialize S3 client."""
        if self._client is not None:
            return  # Already connected
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            # Create session with credentials
            session_kwargs = {
                'region_name': self.region
            }
            
            if self.aws_access_key_id:
                session_kwargs['aws_access_key_id'] = self.aws_access_key_id
            if self.aws_secret_access_key:
                session_kwargs['aws_secret_access_key'] = self.aws_secret_access_key
            if self.aws_session_token:
                session_kwargs['aws_session_token'] = self.aws_session_token
            
            self._session = boto3.Session(**session_kwargs)
            
            # Create S3 client
            client_kwargs = {}
            if self.endpoint_url:
                client_kwargs['endpoint_url'] = self.endpoint_url
            
            self._client = self._session.client('s3', **client_kwargs)
            
            # Test connection by checking bucket exists
            try:
                self._client.head_bucket(Bucket=self.bucket)
                logger.info(f"Connected to S3 bucket: {self.bucket}")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    raise RuntimeError(f"S3 bucket '{self.bucket}' does not exist")
                elif error_code == '403':
                    raise RuntimeError(f"Access denied to S3 bucket '{self.bucket}'")
                else:
                    raise RuntimeError(f"S3 connection error: {e}")
                    
        except ImportError:
            raise RuntimeError("boto3 not installed. Run: pip install boto3")
        except NoCredentialsError:
            raise RuntimeError("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables or use IAM roles.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to S3: {e}")
    
    async def disconnect(self):
        """Close S3 connection."""
        if self._client:
            # boto3 client doesn't need explicit closing
            self._client = None
            self._session = None
            logger.info("Disconnected from S3")
    
    async def list_objects(
        self, 
        prefix: str = "",
        max_keys: int = 1000,
        focus: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List objects in the S3 bucket.
        
        Args:
            prefix: Object key prefix filter
            max_keys: Maximum number of objects to return
            focus: List of object attributes to focus on
            
        Returns:
            List of object metadata dictionaries
            
        Example:
            objects = await s3.list_objects(prefix="data/", focus=["Key", "Size"])
        """
        if self._client is None:
            await self.connect()
        
        try:
            response = self._client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = response.get('Contents', [])
            
            # Apply focus system if specified
            if focus:
                filtered_objects = []
                for obj in objects:
                    filtered_obj = {key: obj.get(key) for key in focus if key in obj}
                    filtered_objects.append(filtered_obj)
                return filtered_objects
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise RuntimeError(f"S3 list_objects failed: {e}")
    
    async def get_object(
        self, 
        key: str, 
        format: str = "auto",
        focus: Optional[List[str]] = None
    ) -> Union[bytes, str, Dict[str, Any], Any]:
        """
        Get an object from S3 with automatic format detection.
        
        Args:
            key: S3 object key
            format: Format type ('auto', 'bytes', 'text', 'json', 'csv', 'pandas')
            focus: List of columns to focus on (for pandas/csv)
            
        Returns:
            Object data in requested format
            
        Example:
            data = await s3.get_object("reports/monthly.csv", format="pandas")
        """
        if self._client is None:
            await self.connect()
        
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            content = response['Body'].read()
            
            # Auto-detect format from file extension
            if format == "auto":
                format = self._detect_format(key)
            
            # Process based on format
            if format == "bytes":
                return content
            elif format == "text":
                return content.decode('utf-8')
            elif format == "json":
                import json
                return json.loads(content.decode('utf-8'))
            elif format == "csv":
                import csv
                content_str = content.decode('utf-8')
                reader = csv.DictReader(io.StringIO(content_str))
                rows = list(reader)
                
                # Apply focus system
                if focus:
                    filtered_rows = []
                    for row in rows:
                        filtered_row = {col: row.get(col) for col in focus if col in row}
                        filtered_rows.append(filtered_row)
                    return filtered_rows
                return rows
            elif format == "pandas":
                try:
                    import pandas as pd
                    
                    # Detect file type for pandas
                    if key.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(content))
                    elif key.endswith('.json'):
                        df = pd.read_json(io.BytesIO(content))
                    elif key.endswith('.parquet'):
                        df = pd.read_parquet(io.BytesIO(content))
                    elif key.endswith('.xlsx'):
                        df = pd.read_excel(io.BytesIO(content))
                    else:
                        # Try CSV as default
                        df = pd.read_csv(io.BytesIO(content))
                    
                    # Apply focus system
                    if focus:
                        available_cols = [col for col in focus if col in df.columns]
                        if available_cols:
                            df = df[available_cols]
                    
                    return df
                except ImportError:
                    raise RuntimeError("pandas not installed. Run: pip install pandas")
            else:
                return content
                
        except Exception as e:
            logger.error(f"Failed to get S3 object {key}: {e}")
            raise RuntimeError(f"S3 get_object failed: {e}")
    
    async def put_object(
        self, 
        key: str, 
        data: Union[bytes, str, Dict[str, Any], Any],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Put an object to S3.
        
        Args:
            key: S3 object key
            data: Data to upload (bytes, string, dict, or pandas DataFrame)
            content_type: Content type (auto-detected if not provided)
            metadata: Object metadata
            
        Returns:
            Upload result metadata
            
        Example:
            result = await s3.put_object("data/output.json", {"result": "success"})
        """
        if self._client is None:
            await self.connect()
        
        try:
            # Process data based on type
            if hasattr(data, 'to_csv'):  # pandas DataFrame
                buffer = io.StringIO()
                data.to_csv(buffer, index=False)
                body = buffer.getvalue().encode('utf-8')
                content_type = content_type or 'text/csv'
            elif hasattr(data, 'to_json'):  # pandas DataFrame to JSON
                buffer = io.StringIO()
                data.to_json(buffer, orient='records', indent=2)
                body = buffer.getvalue().encode('utf-8')
                content_type = content_type or 'application/json'
            elif isinstance(data, dict):
                import json
                body = json.dumps(data, indent=2).encode('utf-8')
                content_type = content_type or 'application/json'
            elif isinstance(data, str):
                body = data.encode('utf-8')
                content_type = content_type or 'text/plain'
            elif isinstance(data, bytes):
                body = data
                content_type = content_type or 'application/octet-stream'
            else:
                # Try to convert to string
                body = str(data).encode('utf-8')
                content_type = content_type or 'text/plain'
            
            # Auto-detect content type from key if not provided
            if not content_type:
                content_type = self._detect_content_type(key)
            
            # Prepare put_object arguments
            put_args = {
                'Bucket': self.bucket,
                'Key': key,
                'Body': body,
                'ContentType': content_type
            }
            
            if metadata:
                put_args['Metadata'] = metadata
            
            # Upload object
            response = self._client.put_object(**put_args)
            
            result = {
                'key': key,
                'etag': response['ETag'],
                'size': len(body),
                'content_type': content_type
            }
            
            if metadata:
                result['metadata'] = metadata
            
            logger.info(f"Uploaded S3 object: {key} ({len(body)} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to put S3 object {key}: {e}")
            raise RuntimeError(f"S3 put_object failed: {e}")
    
    async def upload_dataframe(
        self, 
        df: Any, 
        key: str, 
        format: str = "csv",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Upload a pandas DataFrame to S3.
        
        Args:
            df: pandas DataFrame
            key: S3 object key
            format: Output format ('csv', 'json', 'parquet')
            **kwargs: Additional format-specific parameters
            
        Returns:
            Upload result metadata
            
        Example:
            result = await s3.upload_dataframe(df, "processed/results.parquet", format="parquet")
        """
        if self._client is None:
            await self.connect()
        
        try:
            if format == "csv":
                buffer = io.StringIO()
                df.to_csv(buffer, index=False, **kwargs)
                body = buffer.getvalue().encode('utf-8')
                content_type = 'text/csv'
            elif format == "json":
                buffer = io.StringIO()
                df.to_json(buffer, orient='records', indent=2, **kwargs)
                body = buffer.getvalue().encode('utf-8')
                content_type = 'application/json'
            elif format == "parquet":
                buffer = io.BytesIO()
                df.to_parquet(buffer, **kwargs)
                body = buffer.getvalue()
                content_type = 'application/octet-stream'
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Upload using put_object
            return await self.put_object(key, body, content_type)
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to S3: {e}")
            raise RuntimeError(f"S3 upload_dataframe failed: {e}")
    
    async def download_file(self, key: str, local_path: str) -> str:
        """
        Download an S3 object to local file.
        
        Args:
            key: S3 object key
            local_path: Local file path
            
        Returns:
            Local file path
            
        Example:
            path = await s3.download_file("data/input.csv", "/tmp/input.csv")
        """
        if self._client is None:
            await self.connect()
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self._client.download_file(self.bucket, key, local_path)
            
            logger.info(f"Downloaded S3 object {key} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download S3 object {key}: {e}")
            raise RuntimeError(f"S3 download_file failed: {e}")
    
    async def upload_file(self, local_path: str, key: str) -> Dict[str, Any]:
        """
        Upload a local file to S3.
        
        Args:
            local_path: Local file path
            key: S3 object key
            
        Returns:
            Upload result metadata
            
        Example:
            result = await s3.upload_file("/tmp/output.csv", "results/output.csv")
        """
        if self._client is None:
            await self.connect()
        
        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            # Get file size
            file_size = os.path.getsize(local_path)
            
            # Auto-detect content type
            content_type = self._detect_content_type(local_path)
            
            # Upload file
            self._client.upload_file(
                local_path, 
                self.bucket, 
                key,
                ExtraArgs={'ContentType': content_type}
            )
            
            # Get object metadata
            response = self._client.head_object(Bucket=self.bucket, Key=key)
            
            result = {
                'key': key,
                'etag': response['ETag'],
                'size': file_size,
                'content_type': content_type,
                'local_path': local_path
            }
            
            logger.info(f"Uploaded file {local_path} to S3 object {key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload file {local_path} to S3: {e}")
            raise RuntimeError(f"S3 upload_file failed: {e}")
    
    async def delete_object(self, key: str) -> Dict[str, Any]:
        """
        Delete an object from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            Delete result metadata
            
        Example:
            result = await s3.delete_object("temp/old_file.txt")
        """
        if self._client is None:
            await self.connect()
        
        try:
            response = self._client.delete_object(Bucket=self.bucket, Key=key)
            
            result = {
                'key': key,
                'deleted': True
            }
            
            logger.info(f"Deleted S3 object: {key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete S3 object {key}: {e}")
            raise RuntimeError(f"S3 delete_object failed: {e}")
    
    async def copy_object(
        self, 
        source_key: str, 
        dest_key: str, 
        source_bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Copy an object within S3.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            source_bucket: Source bucket (uses same bucket if not provided)
            
        Returns:
            Copy result metadata
            
        Example:
            result = await s3.copy_object("data/input.csv", "backup/input.csv")
        """
        if self._client is None:
            await self.connect()
        
        try:
            source_bucket = source_bucket or self.bucket
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            
            response = self._client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket,
                Key=dest_key
            )
            
            result = {
                'source_key': source_key,
                'dest_key': dest_key,
                'source_bucket': source_bucket,
                'dest_bucket': self.bucket,
                'etag': response['CopyObjectResult']['ETag']
            }
            
            logger.info(f"Copied S3 object {source_key} to {dest_key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to copy S3 object {source_key}: {e}")
            raise RuntimeError(f"S3 copy_object failed: {e}")
    
    def _detect_format(self, key: str) -> str:
        """Detect format from file extension."""
        ext = Path(key).suffix.lower()
        
        format_map = {
            '.json': 'json',
            '.csv': 'csv',
            '.txt': 'text',
            '.parquet': 'pandas',
            '.xlsx': 'pandas',
            '.xls': 'pandas'
        }
        
        return format_map.get(ext, 'bytes')
    
    def _detect_content_type(self, key: str) -> str:
        """Detect content type from file extension."""
        ext = Path(key).suffix.lower()
        
        content_type_map = {
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.txt': 'text/plain',
            '.parquet': 'application/octet-stream',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif'
        }
        
        return content_type_map.get(ext, 'application/octet-stream')
    
    def get_tools(self) -> List['AgentTool']:
        """
        Expose S3 operations as agent tools.

        Returns:
            List of AgentTool instances for S3 operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="read_s3_file",
                description="Read and parse a file from S3 bucket. Automatically detects format (CSV, JSON, Parquet, text) based on file extension.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "S3 object key (file path within the bucket)"
                        },
                        "format": {
                            "type": "string",
                            "description": "Format hint: 'auto', 'csv', 'json', 'pandas', 'text'. Default is 'auto' which detects from extension."
                        }
                    },
                    "required": ["key"]
                },
                handler=self._tool_read_file,
                category="storage",
                source="plugin",
                plugin_name="S3",
                timeout_seconds=120
            ),
            AgentTool(
                name="write_s3_file",
                description="Write data to S3 bucket. Accepts dictionaries (saved as JSON), strings, or binary data.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "S3 object key (file path within the bucket)"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data to write (dict for JSON, string for text, or bytes for binary)"
                        }
                    },
                    "required": ["key", "data"]
                },
                handler=self._tool_write_file,
                category="storage",
                source="plugin",
                plugin_name="S3",
                timeout_seconds=120
            ),
            AgentTool(
                name="list_s3_objects",
                description="List objects in S3 bucket with optional prefix filter to narrow down results",
                parameters={
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "Filter objects by prefix (folder path). Leave empty to list all objects."
                        },
                        "max_keys": {
                            "type": "integer",
                            "description": "Maximum number of objects to return. Default is 100."
                        }
                    },
                    "required": []
                },
                handler=self._tool_list_objects,
                category="storage",
                source="plugin",
                plugin_name="S3",
                timeout_seconds=60
            ),
            AgentTool(
                name="delete_s3_file",
                description="Delete a file from S3 bucket. This operation is permanent and cannot be undone.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "S3 object key (file path) to delete"
                        }
                    },
                    "required": ["key"]
                },
                handler=self._tool_delete_file,
                category="storage",
                source="plugin",
                plugin_name="S3",
                timeout_seconds=30
            )
        ]

    async def _tool_read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for read_s3_file"""
        key = args.get("key")
        format_hint = args.get("format", "auto")

        data = await self.get_object(key, format=format_hint)

        return {
            "success": True,
            "key": key,
            "data": data,
            "format": self._detect_format(key),
            "bucket": self.bucket
        }

    async def _tool_write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for write_s3_file"""
        key = args.get("key")
        data = args.get("data")

        result = await self.put_object(key, data)

        return {
            "success": True,
            "key": key,
            "size": result.get("size"),
            "location": f"s3://{self.bucket}/{key}",
            "bucket": self.bucket
        }

    async def _tool_list_objects(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_s3_objects"""
        prefix = args.get("prefix", "")
        max_keys = args.get("max_keys", 100)

        objects = await self.list_objects(prefix=prefix, max_keys=max_keys)

        # Simplify object metadata for LLM consumption
        simplified = [
            {
                "key": obj["Key"],
                "size": obj["Size"],
                "modified": str(obj.get("LastModified", ""))
            }
            for obj in objects
        ]

        return {
            "success": True,
            "objects": simplified,
            "count": len(simplified),
            "bucket": self.bucket,
            "prefix": prefix if prefix else "(all objects)"
        }

    async def _tool_delete_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for delete_s3_file"""
        key = args.get("key")

        result = await self.delete_object(key)

        return {
            "success": True,
            "key": key,
            "deleted": result.get("deleted", True),
            "bucket": self.bucket
        }

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def s3(**kwargs) -> S3Plugin:
    """Create S3 plugin with simplified interface."""
    return S3Plugin(**kwargs)