"""Language detection and the multilingual lexicon the router reads.

WeatherGPT is asked questions in three registers that all mean the same thing:

    "How much rain fell yesterday?"
    "कल कितनी बारिश हुई?"
    "kal kitni baarish hui?"

They are one language problem, not three, so the vocabulary lives in one place:
every concept the router needs — a weather variable, a tense marker, a unit of
time — is listed once with its English, Devanagari and romanised forms beside
each other. Adding a term means adding it to the row it belongs to, and both
the temporal parser and the intent detector see it immediately.

Nothing here translates anything. Detection picks a catalog; the numbers in an
answer are always copied from the backend response.
"""

from __future__ import annotations

import re
from typing import Iterable

#: Devanagari, used for Hindi and several other Indian scripts.
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ARABIC = re.compile(r"[؀-ۿ]")

#: Devanagari has no word-boundary that ``\b`` understands: a word may end in a
#: combining mark, which is not a word character, so ``\b`` never matches after
#: it. "कल" (yesterday/tomorrow) is a substring of "कलकत्ता" (Kolkata), and
#: reading that as a date would be a silent, confident error. The boundary that
#: works is "no Devanagari character on either side".
_DEV = "ऀ-ॿ"


def devanagari_word(token: str) -> re.Pattern[str]:
    """A pattern matching ``token`` only as a whole Devanagari word."""
    return re.compile(rf"(?<![{_DEV}]){re.escape(token)}(?![{_DEV}])")


def roman_word(*tokens: str) -> re.Pattern[str]:
    """A pattern matching any of ``tokens`` as whole ASCII words."""
    return re.compile(rf"\b(?:{'|'.join(re.escape(t) for t in tokens)})\b", re.IGNORECASE)


# ---------------------------------------------------------------- detection

#: Romanised Hindi words that carry meaning on their own. One of these is
#: enough to suspect Hinglish; two settle it.
_HINGLISH_STRONG = roman_word(
    "mausam", "mosam", "baarish", "barish", "varsha", "kitna", "kitni", "kitne",
    "kaisa", "kaisi", "kaise", "batao", "bataiye", "bata", "tapman", "taapman",
    "ghante", "ghanta", "ghanto", "pehle", "pahle", "hogi", "hoga", "honge",
    "andhi", "aandhi", "toofan", "garmi", "sardi", "thandi", "dhoop", "nami",
    "sambhavna", "chalegi", "chal", "rahegi", "rahega", "hui", "huyi",
)

#: Function words that are common in Hindi and rare in an English weather
#: question. Individually weak, so they only count towards a total.
_HINGLISH_WEAK = roman_word(
    "aaj", "kal", "parso", "parson", "abhi", "agle", "agla", "agale", "agali",
    "din", "raat", "subah", "sham", "shaam", "dopahar", "hai", "hain", "tha",
    "thi", "the", "kya", "mein", "mai", "mera", "meri", "hawa", "hava",
    "pichle", "pichhle", "beete", "baad", "tak", "wala", "wali", "zyada",
    "jyada", "chahiye", "bahut", "thoda", "kitna",
)

#: Weighted so a single unmistakable word settles it, while a scattering of
#: short function words needs company. Two points is the bar.
_HINGLISH_THRESHOLD = 2


def _score_hinglish(text: str) -> int:
    strong = len(set(m.group(0).lower() for m in _HINGLISH_STRONG.finditer(text)))
    weak = len(set(m.group(0).lower() for m in _HINGLISH_WEAK.finditer(text)))
    return strong * 2 + weak


def script_of(text: str) -> str:
    """The writing system a message is in: ``devanagari``, ``latin`` or ``other``."""
    if DEVANAGARI.search(text):
        return "devanagari"
    if ARABIC.search(text):
        return "other"
    return "latin"


def detect_language(text: str, *, hint: str | None = None, previous: str = "en") -> str:
    """The language to answer this turn in.

    The message itself outranks the language selector. A person who has the
    interface set to English and types a Hindi sentence has asked in Hindi, and
    answering that in English is the wrong answer to the right question —
    which is also what makes language a property of the turn rather than of the
    session.

    The selector still decides every turn the text does not settle, so choosing
    Hindi and then typing an unmarked message keeps answering in Hindi.
    """
    if DEVANAGARI.search(text):
        return "hi"
    if _score_hinglish(text) >= _HINGLISH_THRESHOLD:
        return "hi"
    if hint:
        return hint.strip().replace("_", "-")[:16]
    return previous or "en"


def is_hinglish(text: str) -> bool:
    """True when a Latin-script message is really Hindi."""
    return not DEVANAGARI.search(text) and _score_hinglish(text) >= _HINGLISH_THRESHOLD


# ------------------------------------------------------------------ lexicon


class Term:
    """One concept, written the three ways a person might write it."""

    __slots__ = ("english", "devanagari", "roman", "_patterns")

    def __init__(
        self,
        english: Iterable[str] = (),
        devanagari: Iterable[str] = (),
        roman: Iterable[str] = (),
    ) -> None:
        self.english = tuple(english)
        self.devanagari = tuple(devanagari)
        self.roman = tuple(roman)
        self._patterns: list[re.Pattern[str]] = []
        if self.english or self.roman:
            self._patterns.append(roman_word(*self.english, *self.roman))
        self._patterns.extend(devanagari_word(token) for token in self.devanagari)

    def search(self, text: str) -> re.Match[str] | None:
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return match
        return None

    def __contains__(self, text: str) -> bool:
        return self.search(text) is not None


#: The measurement a question is about. Used to decide what a verdict should
#: talk about, and which figure to lead an answer with.
VARIABLES: dict[str, Term] = {
    "precipitation": Term(
        english=("rain", "rains", "rained", "raining", "rainfall", "precipitation",
                 "shower", "showers", "drizzle", "downpour", "wet"),
        devanagari=("बारिश", "बरसात", "वर्षा", "बूंदाबांदी", "बरसेगी", "बरसा"),
        roman=("baarish", "barish", "varsha", "barsat", "barsaat", "boondabandi"),
    ),
    "temperature": Term(
        english=("temperature", "hot", "cold", "warm", "cool", "degrees", "celsius",
                 "heat", "chilly", "freezing"),
        devanagari=("तापमान", "गर्मी", "ठंड", "ठण्ड", "सर्दी", "गरम", "ठंडा"),
        roman=("tapman", "taapman", "garmi", "sardi", "thand", "thandi", "garam"),
    ),
    "wind": Term(
        english=("wind", "winds", "windy", "gust", "gusts", "gusty", "breeze", "gale"),
        devanagari=("हवा", "वायु", "आंधी", "आँधी", "तूफान", "तूफ़ान"),
        roman=("hawa", "hava", "aandhi", "andhi", "toofan", "tufan"),
    ),
    "humidity": Term(
        english=("humidity", "humid", "muggy", "damp"),
        devanagari=("आर्द्रता", "नमी", "उमस"),
        roman=("aardrata", "nami", "umas"),
    ),
    "cloud": Term(
        english=("cloud", "clouds", "cloudy", "overcast", "sunny", "sunshine", "clear"),
        devanagari=("बादल", "धूप", "साफ"),
        roman=("badal", "baadal", "dhoop", "saaf"),
    ),
    "visibility": Term(
        english=("visibility", "fog", "foggy", "mist", "haze"),
        devanagari=("दृश्यता", "कोहरा", "धुंध"),
        roman=("drishyata", "kohra", "dhundh"),
    ),
    "pressure": Term(english=("pressure", "barometric"), devanagari=("दबाव",), roman=("dabav",)),
    "uv": Term(english=("uv", "ultraviolet", "sunburn"), devanagari=(), roman=()),
    "storm": Term(
        english=("storm", "storms", "stormy", "thunder", "thunderstorm", "lightning",
                 "cyclone", "hail"),
        devanagari=("तूफान", "तूफ़ान", "बिजली", "गरज", "चक्रवात", "ओले"),
        roman=("toofan", "tufan", "bijli", "garaj", "chakravat"),
    ),
}

#: A question is about the weather at all. Used only to tell a weather question
#: from an off-topic one; it never decides tense.
WEATHER_SUBJECT = Term(
    english=("weather", "forecast", "climate", "conditions", "condition", "outside",
             "meteorolog", "met"),
    devanagari=("मौसम", "पूर्वानुमान", "मौसमी"),
    roman=("mausam", "mosam", "purvanuman", "poorvanuman"),
)

#: Existing WeatherGPT rule alerts. Kept intact and untouched by this phase.
ALERT_SUBJECT = Term(
    english=("alert", "alerts", "warning", "warnings", "advisory", "advisories"),
    devanagari=("चेतावनी", "अलर्ट", "सलाह"),
    roman=("chetavni", "chetavani", "alert"),
)

#: Planning questions with no explicit clock reference: "is it safe to travel?"
#:
#: Most of these ask about advisability without ever using the word: "is it
#: worth going out", "is it ok to spray today", "a good time to sail?". They
#: are the ordinary way a person asks, and a detector that reads only "safe"
#: and "risky" sends every one of them to the fallback.
RISK_SUBJECT = Term(
    english=("risk", "risky", "safe", "safety", "advisable", "should i",
             "worth", "worthwhile", "ok to", "okay to", "alright to",
             "good idea", "bad idea", "good time", "bad time", "wise"),
    devanagari=("जोखिम", "सुरक्षित", "खतरा", "ठीक रहेगा", "सही रहेगा",
                "ठीक है क्या", "फायदा"),
    roman=("jokhim", "surakshit", "khatra", "theek rahega", "thik rahega",
           "sahi rahega", "faayda", "fayda"),
)

#: Contexts that change what a useful answer looks like, not what the data says.
PURPOSE_TERMS: dict[str, Term] = {
    # Marine is tested first, and deliberately. A fishing trip is both a "trip"
    # and a voyage; read as travel it gets a road answer, and the variables a
    # skipper needs are not the ones a driver needs.
    "marine": Term(
        english=("boat", "boats", "boating", "sail", "sailing", "sailor", "vessel",
                 "trawler", "catamaran", "dinghy", "kayak", "canoe", "ferry",
                 "fishing", "fisherman", "fishermen", "offshore", "nautical",
                 "harbour", "harbor", "jetty", "quay", "moor", "mooring",
                 "sea", "seas", "at sea", "swell", "tide", "tides", "wave height",
                 "shore", "shoreline", "coast", "coastal", "port"),
        devanagari=("नाव", "नौका", "मछली पकड़ने", "मछुआरा", "समुद्र", "समुद्री",
                    "लहर", "लहरें", "बंदरगाह", "तट"),
        roman=("nav", "nauka", "machhli", "machhuara", "samudra", "samundar",
               "lehar", "leher", "bandargah", "tat"),
    ),
    "agriculture": Term(
        english=("farm", "farming", "crop", "crops", "harvest", "sow", "sowing",
                 "plant", "planted", "planting", "irrigat", "pesticide", "pesticides",
                 "spray", "sprayed", "spraying", "fertiliser", "fertilizer", "field work",
                 "paddy", "chilli", "chili", "cotton", "wheat", "rice"),
        devanagari=("खेत", "खेती", "फसल", "बुवाई", "सिंचाई", "कीटनाशक", "छिड़काव", "उर्वरक"),
        roman=("khet", "kheti", "fasal", "buvai", "sinchai", "keetnashak", "chidkav"),
    ),
    "travel": Term(
        english=("travel", "travelling", "traveling", "trip", "journey", "flight",
                 "flying", "drive", "driving", "commute", "road trip", "voyage"),
        devanagari=("यात्रा", "सफर", "उड़ान", "ड्राइव"),
        roman=("yatra", "safar", "udaan", "safar"),
    ),
    "outdoor_event": Term(
        english=("protest", "protests", "demonstration", "demonstrations", "rally", "rallies",
                 "event", "outdoor", "march", "marching", "gathering", "concert", "festival"),
        devanagari=("प्रदर्शन", "धरना", "रैली", "कार्यक्रम", "सभा"),
        roman=("pradarshan", "dharna", "rally", "karyakram", "sabha"),
    ),
}

#: Words that mean "tell me" and carry no other meaning. Stripped before a
#: place name is read out of a sentence.
IMPERATIVES = roman_word(
    "batao", "bataiye", "bata", "batana", "tell", "show", "give", "please",
)
