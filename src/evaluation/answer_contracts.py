"""Secondary presentation-tolerant explicit answer score, never last-number fallback."""
from decimal import Decimal,InvalidOperation
import re

def verify_formatted_final(text: str, reference: str) -> tuple[bool,str|None]:
    # Strip only paired bold markers around the explicit answer label/value.
    # Match at the end so algebraic expressions and trailing explanations fail.
    m=re.search(r'(?<!\w)(?:\*\*)?Answer:(?:\*\*)?\s*(?:\*\*)?(-?\d+(?:\.\d+)?)(?:\*\*)?\.?\s*$',text,re.I)
    if not m:return False,None
    try:
        a,b=Decimal(m[1]),Decimal(reference.strip())
        return a.is_finite() and b.is_finite() and a==b,m[1]
    except InvalidOperation:return False,m[1]
