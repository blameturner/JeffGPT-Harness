import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from imaplib import IMAP4
from smtplib import SMTP


def send_email(
    to: str,
    subject: str,
    body: str,
    from_addr: str | None = None,
    smtp_host: str = "smtp.example.com",
    smtp_port: int = 587,
    username: str | None = None,
    password: str | None = None,
    use_tls: bool = True,
) -> bool:
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_addr or username
    msg["To"] = to
    msg.attach(MIMEText(body, "plain"))

    with SMTP(smtp_host, smtp_port) as server:
        if use_tls:
            server.starttls()
        if username and password:
            server.login(username, password)
        server.sendmail(from_addr or username or "", to, msg.as_string())
    return True


def fetch_emails(
    imap_host: str = "imap.example.com",
    imap_port: int = 993,
    username: str = "",
    password: str = "",
    mailbox: str = "INBOX",
    subject_filter: str | None = None,
    from_filter: str | None = None,
    unread_only: bool = False,
) -> list[dict]:
    emails = []
    criteria = []
    if unread_only:
        criteria.append("UNSEEN")
    if subject_filter:
        criteria.append(f'SUBJECT "{subject_filter}"')
    if from_filter:
        criteria.append(f'FROM "{from_filter}"')

    with IMAP4(imap_host, imap_port) as server:
        server.login(username, password)
        server.select(mailbox)
        _, message_ids = server.search(None, *(criteria or ["ALL"]))

        for msg_id in message_ids[0].split():
            _, msg_data = server.fetch(msg_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            emails.append({
                "id": msg_id.decode(),
                "subject": msg.get("Subject", ""),
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "date": msg.get("Date", ""),
                "body": get_message_body(msg),
            })
    return emails


def get_message_body(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode()
    return msg.get_payload(decode=True).decode() if msg.get_payload() else ""