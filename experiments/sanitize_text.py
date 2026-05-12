import re


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
CNPJ_RE = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?(?:9\s*)?\d{4}[-.\s]?\d{4}(?!\d)")
LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
ZIP_CODE_RE = re.compile(r"\b\d{5}-?\d{3}\b")
MULTISPACE_RE = re.compile(r"[ \t]+")


def sanitize_text(text: str) -> str:
    sanitized = "" if text is None else str(text)
    sanitized = URL_RE.sub("<URL>", sanitized)
    sanitized = EMAIL_RE.sub("<EMAIL>", sanitized)
    sanitized = CNPJ_RE.sub("<DOCUMENTO>", sanitized)
    sanitized = CPF_RE.sub("<DOCUMENTO>", sanitized)
    sanitized = ZIP_CODE_RE.sub("<CEP>", sanitized)
    sanitized = PHONE_RE.sub("<TELEFONE>", sanitized)
    sanitized = LONG_NUMBER_RE.sub("<NUMERO>", sanitized)
    sanitized = MULTISPACE_RE.sub(" ", sanitized)
    return sanitized.strip()
