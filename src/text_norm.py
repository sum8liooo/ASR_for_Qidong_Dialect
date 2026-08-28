# -*- coding: utf-8 -*-
"""Chinese text normalisation applied BEFORE CER scoring.

Why this matters (cite in §4.4.1): raw Whisper output mixes full/half-width
punctuation and sometimes traditional characters; scoring without a fixed
normalisation policy can swing CER by several points and makes conditions
non-comparable. The SAME function must be applied to references and hypotheses.
"""
import re
import unicodedata

# characters to strip entirely (punctuation, spaces)
_PUNCT = re.compile(
    "[\\s，。！？、；：""''《》〈〉（）()【】\\[\\]—…·,.!?;:'\"<>" + "“”‘’~～-]+"
)


def normalize(text: str) -> str:
    """Normalisation pipeline: NFKC -> strip punctuation/space -> lowercase latin."""
    if text is None:
        return ""
    # 1. Unicode NFKC: full-width digits/latin -> half-width, etc.
    text = unicodedata.normalize("NFKC", text)
    # 2. remove all punctuation and whitespace (CER is character-level on hanzi)
    text = _PUNCT.sub("", text)
    # 3. lowercase any residual latin (e.g. code-switched words)
    text = text.lower()
    return text


if __name__ == "__main__":
    demo = "啟東話，ＡＳＲ 测试！！（第 1 句）"
    print(normalize(demo))  # -> 啟東話asr测试第1句  (note: trad->simp NOT applied here;
    # decide with supervisor whether to add OpenCC t2s conversion as step 0)
