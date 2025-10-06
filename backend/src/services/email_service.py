"""
Real email service for sending alerts via SMTP
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
import logging

class EmailService:
    def __init__(self):
        # Gmail SMTP configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "zombiewins23@gmail.com"
        # You need to set your Gmail App Password here
        # To get App Password: Gmail Settings > Security > 2-Step Verification > App passwords
        # Replace this with your actual 16-character App Password
        self.sender_password = "your_gmail_app_password_here"
        
    def send_alert(self, recipient: str, subject: str, message: str) -> Dict[str, Any]:
        """
        Send email alert via Gmail SMTP
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(message, 'plain'))
            
            # Create SMTP session
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                
                # Send email
                text = msg.as_string()
                server.sendmail(self.sender_email, recipient, text)
                
            logging.getLogger(__name__).info(f"Email sent successfully to {recipient}")
            return {
                "success": True,
                "message": "Email sent successfully",
                "recipient": recipient
            }
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Email sending failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Email sending failed"
            }

# For testing without real SMTP
class MockEmailService:
    def __init__(self):
        self.sent_emails = []
        
    def send_alert(self, recipient: str, subject: str, message: str) -> Dict[str, Any]:
        """
        Mock email service that logs emails instead of sending them
        """
        email_data = {
            "recipient": recipient,
            "subject": subject,
            "message": message,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        self.sent_emails.append(email_data)
        
        print(f"📧 MOCK EMAIL SENT:")
        print(f"   To: {recipient}")
        print(f"   Subject: {subject}")
        print(f"   Message: {message[:100]}...")
        
        return {
            "success": True,
            "message": "Mock email sent successfully",
            "recipient": recipient,
            "mock": True
        }
    
    def get_sent_emails(self):
        """Get list of sent emails for testing"""
        return self.sent_emails
