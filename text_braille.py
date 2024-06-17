import re
from jamo import h2j, j2hcj

# 한글 자음과 모음에 대한 점자 매핑 테이블
braille_map = {
    'ᄀ': '⠁', 'ᄁ': '⠈', 'ᄂ': '⠂', 'ᄃ': '⠄', 'ᄄ': '⠔', 'ᄅ': '⠈', 'ᄆ': '⠘', 'ᄇ': '⠢', 'ᄈ': '⠢', 'ᄉ': '⠔', 'ᄊ': '⠴', 'ᄋ': '⠶', 'ᄌ': '⠦', 'ᄍ': '⠦', 'ᄎ': '⠲',
    'ᄏ': '⠜', 'ᄐ': '⠼', 'ᄑ': '⠳', 'ᄒ': '⠿',
    'ᅡ': '⠗', 'ᅣ': '⠚', 'ᅥ': '⠜', 'ᅧ': '⠣', 'ᅩ': '⠹', 'ᅭ': '⠧', 'ᅮ': '⠏', 'ᅲ': '⠟', 'ᅳ': '⠍', 'ᅵ': '⠩',
    'ᅢ': '⠙', 'ᅤ': '⠋', 'ᅦ': '⠌', 'ᅨ': '⠓', 'ᅴ': '⠥',
    'ᆨ': '⠁', 'ᆩ': '⠈', 'ᆫ': '⠂', 'ᆮ': '⠄', 'ᆯ': '⠈', 'ᆷ': '⠘', 'ᆸ': '⠢', 'ᆺ': '⠔', 'ᆻ': '⠔⠔', 'ᆼ': '⠶', 'ᆽ': '⠦', 'ᆾ': '⠲',
    'ᆿ': '⠜', 'ᇀ': '⠼', 'ᇁ': '⠳', 'ᇂ': '⠿',
    # 복합 받침
    'ᆰ': '⠁⠈', 'ᆱ': '⠁⠘', 'ᆲ': '⠁⠢', 'ᆳ': '⠁⠔', 'ᆴ': '⠁⠲', 'ᆵ': '⠁⠳', 'ᆶ': '⠁⠿',
    'ᆹ': '⠢⠔', 'ᆺᇂ': '⠔⠿'
}

def decompose_hangul_char(char):
    """ 한글 음절을 초성, 중성, 종성으로 분해 """
    base_code, cho_code, jung_code = 44032, 588, 28
    chosung_list = [chr(code) for code in range(0x1100, 0x1113)]
    jungsung_list = [chr(code) for code in range(0x1161, 0x1176)]
    jongsung_list = [chr(code) for code in range(0x11A8, 0x11C3)]

    if '가' <= char <= '힣':
        code = ord(char) - base_code
        cho = code // cho_code
        jung = (code - cho * cho_code) // jung_code
        jong = (code - cho * cho_code - jung * jung_code)

        return chosung_list[cho], jungsung_list[jung], (jongsung_list[jong - 1] if jong > 0 else '')
    return char

def hangul_to_braille(hangul_text):
    braille_text = ''
    for char in hangul_text:
        decomposed = decompose_hangul_char(char)
        if isinstance(decomposed, tuple):
            for jamo in decomposed:
                if jamo in braille_map:
                    braille_text += braille_map[jamo]
                else:
                    braille_text += jamo
        else:
            braille_text += decomposed
    return braille_text

# 테스트
hangul_text = "이 캐릭터는 노란색 모자와 옷을 입은 캐릭터입니다"
braille_text = hangul_to_braille(hangul_text)
print("원본 한글:", hangul_text)
print("점자 변환:", braille_text)


