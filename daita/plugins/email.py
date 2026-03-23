"""
Email plugin for Daita Agents.

Universal email integration supporting IMAP/SMTP protocols for reading and sending emails.
Works with Gmail, Outlook, Yahoo, and any email provider supporting standard protocols.
"""

import asyncio
import logging
import os
import email as email_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
from .base import BasePlugin

if TYPE_CHECKING:
    from ..core.tools import AgentTool

logger = logging.getLogger(__name__)


# Provider presets for common email services
EMAIL_PROVIDERS = {
    "gmail": {
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
    },
    "outlook": {
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "smtp_host": "smtp.office365.com",
        "smtp_port": 587,
        "use_tls": True,
    },
    "yahoo": {
        "imap_host": "imap.mail.yahoo.com",
        "imap_port": 993,
        "smtp_host": "smtp.mail.yahoo.com",
        "smtp_port": 587,
        "use_tls": True,
    },
    "icloud": {
        "imap_host": "imap.mail.me.com",
        "imap_port": 993,
        "smtp_host": "smtp.mail.me.com",
        "smtp_port": 587,
        "use_tls": True,
    },
}


class EmailPlugin(BasePlugin):
    """
    Universal email plugin for agents supporting IMAP/SMTP protocols.

    Works with Gmail, Outlook, Yahoo, iCloud, and any email provider
    supporting standard IMAP/SMTP protocols.
    """

    def __init__(
        self,
        email_address: str,
        password: str,
        provider: Optional[str] = None,
        imap_host: Optional[str] = None,
        imap_port: int = 993,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        use_tls: bool = True,
        **kwargs,
    ):
        """
        Initialize email connection.

        Args:
            email_address: Your email address
            password: Email password or app-specific password
            provider: Provider preset ("gmail", "outlook", "yahoo", "icloud") - auto-configures IMAP/SMTP
            imap_host: IMAP server host (overrides provider preset)
            imap_port: IMAP server port (default: 993)
            smtp_host: SMTP server host (overrides provider preset)
            smtp_port: SMTP server port (default: 587)
            use_tls: Use TLS for SMTP (default: True)
            **kwargs: Additional configuration options

        Examples:
            # Using provider preset
            email = EmailPlugin(
                email_address="user@gmail.com",
                password="app_password",
                provider="gmail"
            )

            # Custom IMAP/SMTP configuration
            email = EmailPlugin(
                email_address="user@company.com",
                password="password",
                imap_host="mail.company.com",
                smtp_host="mail.company.com"
            )
        """
        if not email_address or not email_address.strip():
            raise ValueError("email_address cannot be empty")

        if not password or not password.strip():
            raise ValueError("password cannot be empty")

        self.email_address = email_address
        self.password = password

        # Configure based on provider preset or custom settings
        if provider:
            if provider.lower() not in EMAIL_PROVIDERS:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Supported providers: {', '.join(EMAIL_PROVIDERS.keys())}"
                )
            preset = EMAIL_PROVIDERS[provider.lower()]
            self.imap_host = imap_host or preset["imap_host"]
            self.imap_port = imap_port if imap_host else preset["imap_port"]
            self.smtp_host = smtp_host or preset["smtp_host"]
            self.smtp_port = smtp_port if smtp_host else preset["smtp_port"]
            self.use_tls = preset["use_tls"]
        else:
            if not imap_host or not smtp_host:
                raise ValueError(
                    "Either 'provider' or both 'imap_host' and 'smtp_host' must be specified"
                )
            self.imap_host = imap_host
            self.imap_port = imap_port
            self.smtp_host = smtp_host
            self.smtp_port = smtp_port
            self.use_tls = use_tls

        self._imap = None
        self._smtp = None

        logger.debug(
            f"Email plugin configured for {email_address} "
            f"(IMAP: {self.imap_host}:{self.imap_port}, SMTP: {self.smtp_host}:{self.smtp_port})"
        )

    async def connect(self):
        """Connect to email server (IMAP and SMTP)."""
        if self._imap is not None and self._smtp is not None:
            return  # Already connected

        try:
            import imaplib
            import smtplib

            imap_host, imap_port = self.imap_host, self.imap_port
            smtp_host, smtp_port = self.smtp_host, self.smtp_port
            email_address, password, use_tls = (
                self.email_address,
                self.password,
                self.use_tls,
            )

            def _connect():
                logger.debug(f"Connecting to IMAP server: {imap_host}:{imap_port}")
                imap = imaplib.IMAP4_SSL(imap_host, imap_port)
                imap.login(email_address, password)

                logger.debug(f"Connecting to SMTP server: {smtp_host}:{smtp_port}")
                smtp = smtplib.SMTP(smtp_host, smtp_port)
                if use_tls:
                    smtp.starttls()
                smtp.login(email_address, password)
                return imap, smtp

            loop = asyncio.get_running_loop()
            self._imap, self._smtp = await loop.run_in_executor(None, _connect)
            logger.info(f"Connected to email account: {self.email_address}")

        except ImportError:
            raise RuntimeError(
                "Email libraries not available. They should be part of Python standard library."
            )
        except Exception as e:
            error_msg = str(e)
            troubleshooting = []

            if "authentication failed" in error_msg.lower():
                troubleshooting.append(
                    "Authentication failed. For Gmail/Outlook, use app-specific passwords, not your regular password."
                )
                troubleshooting.append(
                    "Gmail: Enable 'Less secure app access' or create an app password at https://myaccount.google.com/apppasswords"
                )
                troubleshooting.append(
                    "Outlook: Enable IMAP in settings and use app password if 2FA is enabled."
                )

            if troubleshooting:
                enhanced_error = f"{error_msg}\n\nTroubleshooting:\n" + "\n".join(
                    f"  - {tip}" for tip in troubleshooting
                )
                logger.error(enhanced_error)
                raise RuntimeError(enhanced_error)
            else:
                raise RuntimeError(f"Failed to connect to email server: {error_msg}")

    async def disconnect(self):
        """Disconnect from email server."""
        loop = asyncio.get_running_loop()

        if self._imap:
            imap = self._imap
            self._imap = None
            try:
                await loop.run_in_executor(None, imap.logout)
            except Exception as e:
                logger.warning(f"Error closing IMAP connection: {e}")

        if self._smtp:
            smtp = self._smtp
            self._smtp = None
            try:
                await loop.run_in_executor(None, smtp.quit)
            except Exception as e:
                logger.warning(f"Error closing SMTP connection: {e}")

        logger.info("Disconnected from email server")

    async def list_emails(
        self,
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
        search_criteria: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List emails from a folder.

        Args:
            folder: Email folder name (default: "INBOX")
            limit: Maximum number of emails to return
            unread_only: Only return unread emails
            search_criteria: IMAP search criteria (e.g., 'FROM "sender@example.com"')

        Returns:
            List of email summaries with metadata

        Example:
            emails = await email.list_emails(folder="INBOX", limit=5, unread_only=True)
        """
        if self._imap is None:
            await self.connect()

        try:
            imap = self._imap

            def _list():
                imap.select(folder, readonly=True)

                if search_criteria:
                    criteria = search_criteria
                elif unread_only:
                    criteria = "UNSEEN"
                else:
                    criteria = "ALL"

                status, messages = imap.search(None, criteria)
                if status != "OK":
                    raise RuntimeError(f"Failed to search emails: {status}")

                email_ids = messages[0].split()
                email_ids = email_ids[-limit:] if len(email_ids) > limit else email_ids
                email_ids = list(reversed(email_ids))

                emails = []
                for email_id in email_ids:
                    status, msg_data = imap.fetch(email_id, "(RFC822.HEADER)")
                    if status != "OK":
                        continue

                    raw_email = msg_data[0][1]
                    msg = email_lib.message_from_bytes(raw_email)

                    email_info = {
                        "id": email_id.decode(),
                        "subject": msg.get("Subject", ""),
                        "from": msg.get("From", ""),
                        "to": msg.get("To", ""),
                        "date": msg.get("Date", ""),
                        "has_attachments": False,
                    }

                    if email_info["date"]:
                        try:
                            email_info["date_parsed"] = parsedate_to_datetime(
                                email_info["date"]
                            ).isoformat()
                        except Exception:
                            email_info["date_parsed"] = None

                    emails.append(email_info)

                return emails

            loop = asyncio.get_running_loop()
            emails = await loop.run_in_executor(None, _list)
            logger.info(f"Listed {len(emails)} emails from {folder}")
            return emails

        except Exception as e:
            logger.error(f"Failed to list emails: {e}")
            raise RuntimeError(f"Email list_emails failed: {e}")

    async def read_email(
        self, email_id: str, folder: str = "INBOX", mark_as_read: bool = False
    ) -> Dict[str, Any]:
        """
        Read full email content by ID.

        Args:
            email_id: Email ID from list_emails()
            folder: Email folder name (default: "INBOX")
            mark_as_read: Mark email as read after fetching

        Returns:
            Full email content with body and metadata

        Example:
            email = await email.read_email(email_id="12345")
        """
        if self._imap is None:
            await self.connect()

        try:
            imap = self._imap

            def _read():
                readonly = not mark_as_read
                imap.select(folder, readonly=readonly)

                status, msg_data = imap.fetch(email_id.encode(), "(RFC822)")
                if status != "OK":
                    raise RuntimeError(f"Failed to fetch email: {status}")

                raw_email = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw_email)

                body = ""
                body_html = ""
                attachments = []

                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition", ""))

                        if (
                            content_type == "text/plain"
                            and "attachment" not in content_disposition
                        ):
                            body = part.get_payload(decode=True).decode(errors="ignore")
                        elif (
                            content_type == "text/html"
                            and "attachment" not in content_disposition
                        ):
                            body_html = part.get_payload(decode=True).decode(
                                errors="ignore"
                            )
                        elif "attachment" in content_disposition:
                            filename = part.get_filename()
                            if filename:
                                attachments.append(
                                    {
                                        "filename": filename,
                                        "content_type": content_type,
                                        "size": len(part.get_payload(decode=True)),
                                    }
                                )
                else:
                    content_type = msg.get_content_type()
                    if content_type == "text/plain":
                        body = msg.get_payload(decode=True).decode(errors="ignore")
                    elif content_type == "text/html":
                        body_html = msg.get_payload(decode=True).decode(errors="ignore")

                email_data = {
                    "id": email_id,
                    "subject": msg.get("Subject", ""),
                    "from": msg.get("From", ""),
                    "to": msg.get("To", ""),
                    "cc": msg.get("Cc", ""),
                    "date": msg.get("Date", ""),
                    "body": body,
                    "body_html": body_html,
                    "attachments": attachments,
                }

                if email_data["date"]:
                    try:
                        email_data["date_parsed"] = parsedate_to_datetime(
                            email_data["date"]
                        ).isoformat()
                    except Exception:
                        email_data["date_parsed"] = None

                if mark_as_read:
                    imap.store(email_id.encode(), "+FLAGS", "\\Seen")

                return email_data

            loop = asyncio.get_running_loop()
            email_data = await loop.run_in_executor(None, _read)
            logger.info(f"Read email: {email_data['subject']}")
            return email_data

        except Exception as e:
            logger.error(f"Failed to read email: {e}")
            raise RuntimeError(f"Email read_email failed: {e}")

    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        html: bool = False,
        attachments: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Send an email.

        Args:
            to: Recipient email address(es)
            subject: Email subject
            body: Email body (plain text or HTML)
            cc: CC recipient(s)
            bcc: BCC recipient(s)
            html: Whether body is HTML (default: False)
            attachments: List of file paths to attach

        Returns:
            Send result with metadata

        Example:
            result = await email.send_email(
                to="recipient@example.com",
                subject="Test Email",
                body="This is a test email"
            )
        """
        if self._smtp is None:
            await self.connect()

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.email_address

            # Handle recipients
            if isinstance(to, list):
                msg["To"] = ", ".join(to)
                recipients = to
            else:
                msg["To"] = to
                recipients = [to]

            if cc:
                if isinstance(cc, list):
                    msg["Cc"] = ", ".join(cc)
                    recipients.extend(cc)
                else:
                    msg["Cc"] = cc
                    recipients.append(cc)

            if bcc:
                if isinstance(bcc, list):
                    recipients.extend(bcc)
                else:
                    recipients.append(bcc)

            msg["Subject"] = subject
            msg["Date"] = email_lib.utils.formatdate(localtime=True)

            # Attach body
            body_type = "html" if html else "plain"
            msg.attach(MIMEText(body, body_type))

            # Attach files
            if attachments:
                for file_path in attachments:
                    if not os.path.exists(file_path):
                        logger.warning(f"Attachment not found: {file_path}")
                        continue

                    with open(file_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())

                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(file_path)}",
                    )
                    msg.attach(part)

            # Send email (synchronous — wrapped in executor)
            msg_str = msg.as_string()
            smtp = self._smtp
            sender = self.email_address
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: smtp.sendmail(sender, recipients, msg_str)
            )

            result = {
                "to": recipients,
                "subject": subject,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Sent email to {recipients}: {subject}")
            return result

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise RuntimeError(f"Email send_email failed: {e}")

    async def reply_to_email(
        self,
        email_id: str,
        body: str,
        folder: str = "INBOX",
        html: bool = False,
        attachments: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Reply to an email.

        Args:
            email_id: ID of email to reply to
            body: Reply body
            folder: Folder containing the email
            html: Whether body is HTML
            attachments: Optional file attachments

        Returns:
            Send result

        Example:
            result = await email.reply_to_email(
                email_id="12345",
                body="Thank you for your email!"
            )
        """
        # Read original email
        original = await self.read_email(email_id, folder=folder)

        # Extract reply address
        reply_to = original.get("from")
        subject = original.get("subject", "")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"

        # Send reply
        return await self.send_email(
            to=reply_to, subject=subject, body=body, html=html, attachments=attachments
        )

    async def search_emails(
        self, query: str, folder: str = "INBOX", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search emails using IMAP search criteria.

        Args:
            query: IMAP search query (e.g., 'FROM "sender@example.com"', 'SUBJECT "meeting"')
            folder: Folder to search
            limit: Maximum results

        Returns:
            List of matching emails

        Example:
            emails = await email.search_emails('FROM "boss@company.com"')
        """
        return await self.list_emails(folder=folder, limit=limit, search_criteria=query)

    async def delete_email(
        self, email_id: str, folder: str = "INBOX", permanent: bool = False
    ) -> Dict[str, Any]:
        """
        Delete an email.

        Args:
            email_id: Email ID to delete
            folder: Folder containing the email
            permanent: Permanently delete (True) or move to trash (False)

        Returns:
            Deletion result

        Example:
            result = await email.delete_email(email_id="12345")
        """
        if self._imap is None:
            await self.connect()

        try:
            imap = self._imap

            def _delete():
                imap.select(folder, readonly=False)
                imap.store(email_id.encode(), "+FLAGS", "\\Deleted")
                if permanent:
                    imap.expunge()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _delete)

            logger.info(f"Deleted email: {email_id}")
            return {"email_id": email_id, "permanent": permanent, "deleted": True}

        except Exception as e:
            logger.error(f"Failed to delete email: {e}")
            raise RuntimeError(f"Email delete_email failed: {e}")

    def get_tools(self) -> List["AgentTool"]:
        """
        Expose email operations as agent tools.

        Returns:
            List of AgentTool instances for email operations
        """
        from ..core.tools import AgentTool

        return [
            AgentTool(
                name="list_emails",
                description="List emails from inbox or a specific folder. Returns email summaries with subject, sender, date, and ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": "Email folder name (default: INBOX)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of emails to return (default: 10)",
                        },
                        "unread_only": {
                            "type": "boolean",
                            "description": "Only return unread emails (default: false)",
                        },
                    },
                    "required": [],
                },
                handler=self._tool_list_emails,
                category="email",
                source="plugin",
                plugin_name="Email",
                timeout_seconds=60,
            ),
            AgentTool(
                name="read_email",
                description="Read full email content including body, attachments, and metadata. Use the email ID from list_emails.",
                parameters={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Email ID from list_emails",
                        },
                        "folder": {
                            "type": "string",
                            "description": "Email folder name (default: INBOX)",
                        },
                        "mark_as_read": {
                            "type": "boolean",
                            "description": "Mark email as read after fetching (default: false)",
                        },
                    },
                    "required": ["email_id"],
                },
                handler=self._tool_read_email,
                category="email",
                source="plugin",
                plugin_name="Email",
                timeout_seconds=60,
            ),
            AgentTool(
                name="send_email",
                description="Send an email with optional HTML formatting and attachments.",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject line",
                        },
                        "body": {"type": "string", "description": "Email body content"},
                        "html": {
                            "type": "boolean",
                            "description": "Whether body is HTML formatted (default: false)",
                        },
                        "cc": {
                            "type": "string",
                            "description": "Optional CC recipient email address",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
                handler=self._tool_send_email,
                category="email",
                source="plugin",
                plugin_name="Email",
                timeout_seconds=60,
            ),
            AgentTool(
                name="reply_to_email",
                description="Reply to an existing email. Automatically sets the reply-to address and subject.",
                parameters={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "ID of email to reply to",
                        },
                        "body": {"type": "string", "description": "Reply message body"},
                        "html": {
                            "type": "boolean",
                            "description": "Whether body is HTML formatted (default: false)",
                        },
                    },
                    "required": ["email_id", "body"],
                },
                handler=self._tool_reply_to_email,
                category="email",
                source="plugin",
                plugin_name="Email",
                timeout_seconds=60,
            ),
            AgentTool(
                name="search_emails",
                description="Search emails using IMAP search criteria. Supports searching by sender, subject, date, etc.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "IMAP search query (e.g., 'FROM \"sender@example.com\"', 'SUBJECT \"meeting\"')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                        },
                    },
                    "required": ["query"],
                },
                handler=self._tool_search_emails,
                category="email",
                source="plugin",
                plugin_name="Email",
                timeout_seconds=60,
            ),
        ]

    async def _tool_list_emails(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for list_emails"""
        folder = args.get("folder", "INBOX")
        limit = args.get("limit", 10)
        unread_only = args.get("unread_only", False)

        emails = await self.list_emails(
            folder=folder, limit=limit, unread_only=unread_only
        )

        return {"emails": emails, "count": len(emails)}

    async def _tool_read_email(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for read_email"""
        email_id = args.get("email_id")
        folder = args.get("folder", "INBOX")
        mark_as_read = args.get("mark_as_read", False)

        email_data = await self.read_email(
            email_id=email_id, folder=folder, mark_as_read=mark_as_read
        )

        # Drop body_html — it duplicates body content with HTML markup and bloats the context.
        # Truncate body to avoid very large emails exceeding LLM context limits.
        MAX_BODY_CHARS = 8000
        body = email_data.get("body") or ""
        truncated = len(body) > MAX_BODY_CHARS
        safe_email = {k: v for k, v in email_data.items() if k != "body_html"}
        safe_email["body"] = body[:MAX_BODY_CHARS]
        if truncated:
            safe_email["body_truncated"] = True
            safe_email["body_total_chars"] = len(body)

        return {"email": safe_email}

    @staticmethod
    def _parse_recipients(val) -> List[str]:
        """Accept a comma-separated string or a list of addresses."""
        if isinstance(val, str):
            return [addr.strip() for addr in val.split(",") if addr.strip()]
        return list(val) if val else []

    async def _tool_send_email(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for send_email"""
        to = self._parse_recipients(args.get("to"))
        subject = args.get("subject")
        body = args.get("body")
        html = args.get("html", False)
        cc = self._parse_recipients(args.get("cc")) or None

        return await self.send_email(
            to=to, subject=subject, body=body, html=html, cc=cc
        )

    async def _tool_reply_to_email(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for reply_to_email"""
        email_id = args.get("email_id")
        body = args.get("body")
        html = args.get("html", False)

        return await self.reply_to_email(email_id=email_id, body=body, html=html)

    async def _tool_search_emails(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Tool handler for search_emails"""
        query = args.get("query")
        limit = args.get("limit", 10)

        emails = await self.search_emails(query=query, limit=limit)

        return {"emails": emails, "count": len(emails)}

    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


def email(**kwargs) -> EmailPlugin:
    """Create Email plugin with simplified interface."""
    return EmailPlugin(**kwargs)
