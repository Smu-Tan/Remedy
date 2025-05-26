"""
Language utilities for Remedy MT Evaluation
"""

# Language mapping dictionary from ISO codes to full names
LANG_MAP = {
    'en': 'English', 'zh': 'Chinese', 'fr': 'French', 'de': 'German',
    'es': 'Spanish', 'ru': 'Russian', 'ar': 'Arabic', 'ko': 'Korean',
    'ja': 'Japanese', 'cs': 'Czech', 'lv': 'Latvian', 'fi': 'Finnish',
    'tr': 'Turkish', 'et': 'Estonian', 'lt': 'Lithuanian', 'kk': 'Kazakh',
    'pl': 'Polish', 'ta': 'Tamil', 'bn': 'Bengali', 'hi': 'Hindi',
    'mr': 'Marathi', 'ne': 'Nepali', 'ro': 'Romanian', 'si': 'Sinhala',
    'uk': 'Ukrainian', 'hr': 'croatian', 'liv': 'Livonian', 'sah': 'Yakut',
    'he': 'hebrew', 'gu': 'Gujarati', 'km': 'Khmer', 'ps': 'Pushto',
    'ha': 'Hausa', 'is': 'Icelandic', 'xh': 'Xhosa', 'zu': 'Zulu'
}

def get_full_lang_name(abbr):
    """Convert language abbreviation to full name."""
    if abbr not in LANG_MAP:
        supported_langs = ', '.join(sorted(LANG_MAP.keys()))
        raise ValueError(f"Language code '{abbr}' is not supported. "
                        f"Supported language codes are: {supported_langs}. "
                        f"Please add '{abbr}' to the LANG_MAP dictionary if needed.")
    return LANG_MAP[abbr]

def get_supported_languages():
    """Return a list of supported language codes."""
    return sorted(LANG_MAP.keys())

def is_supported_language(lang_code):
    """Check if a language code is supported."""
    return lang_code in LANG_MAP 