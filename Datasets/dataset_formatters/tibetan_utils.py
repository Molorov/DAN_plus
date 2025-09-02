import unicodedata as ucd




def is_headpos_consonant(letter):
    if ord(letter) >= int('0F40', 16) and ord(letter) <= int('0F69', 16) or \
            ord(letter) == int('0F6A', 16):
        return True
    else:
        return False

def is_subjoined_consonant(letter):
    if ord(letter) >= int('0F90', 16) and ord(letter) <= int('0FB9', 16) or \
            ord(letter) >= int('0FBA', 16) and ord(letter) <= int('0FBC', 16):
        return True
    else:
        return False

def is_vowel(letter):
    if ord(letter) >= int('0F72', 16) and ord(letter) <= int('0F7E', 16) or \
            ord(letter) >= int('0F80', 16) and ord(letter) <= int('0F83', 16):
        return True
    else:
        return False

def top_joined(letter):
    if ord(letter) == int('0F72', 16) or \
        ord(letter) >=  int('0F7A', 16) and ord(letter) <= int('0F7E', 16) or \
         ord(letter) == int('0F80', 16) or \
          ord(letter) >=  int('0F82', 16) and ord(letter) <= int('0F83', 16):
        return True
    return False


correct_map = {
    'འིུ': 'འིུ',
    'ཀིུ': 'ཀིུ'
}

"""用于修正字丁的错误"""
def correct(char):
    if char in correct_map.keys():
        char = correct_map[char]
    return char



def string2char_list(in_str):
    char_list = list()
    char = ''
    for i, letter in enumerate(in_str):
        if is_subjoined_consonant(letter) and len(char) >= 1 and \
                (is_headpos_consonant(char[-1]) or is_subjoined_consonant(char[-1]) or ord(char[-1]) == int('0F39', 16)):
            char += letter
        elif ord(letter) == int('0F39', 16) and len(char) >= 1 and \
                (is_headpos_consonant(char[-1]) or is_subjoined_consonant(char[-1])):
            char += letter
        elif ord(letter) == int('0F71', 16) and len(char) >= 1 and \
                (is_headpos_consonant(char[-1]) or is_subjoined_consonant(char[-1]) or ord(char[-1]) == int('0F39', 16)):
            char += letter
        elif (is_vowel(letter) or ord(letter) == int('0F84', 16)) and len(char) >= 1 and \
                (ord(char[-1]) == int('0F71', 16) or ord(char[-1]) == int('0F39', 16) or \
                    is_headpos_consonant(char[-1]) or is_subjoined_consonant(char[-1]) or is_vowel(char[-1])):
            char += letter
        else:
            if len(char) > 0:
                char = correct(char)
                char_list.append(char)
            """这两个字丁看上去一样，因此做替换"""
            letter = chr(int('0F0B', 16)) if ord(letter) == int('0F0C', 16) else letter
            """把non-breaking space替换成一般空格"""
            letter = ' ' if letter == '\xa0' else letter
            char = letter
    if len(char) > 0:
        char = correct(char)
        char_list.append(char)


    norm_char_list = []
    for char in char_list:
        # if ucd.normalize('NFD', char) in debug_chars:
        #     print('debug')
        norm_char = ucd.normalize('NFD', char)
        norm_char = correct(norm_char)
        norm_char_list.append(norm_char)

    return norm_char_list

def normalize_char(char, debug=False):
    new_char = char.replace(chr(0xF7B), chr(0xF7A)+chr(0xF7A)) # mapping 0xF7A+0xF7A to 0xF7B
    new_char = new_char.replace(chr(0xF00), chr(0xF68) + chr(0xF7C) + chr(0xF7E)) # mapping 0xF00 to 0xF68+0xF7C+0xF7E
    new_char = new_char.replace(chr(0xF7D), chr(0xF7C)+chr(0xF7C)) # mapping 0xF7C+0xF7C to 0xF7D
    if new_char == chr(0xF6A):
        new_char = chr(0xF62)  # repalce single 0xF6A to single 0xF62 //consonant "ra"
    # new_char = new_char.replace(chr(0xF5D), chr(0xF63)+chr(0xFA6)) # mapping 0xF5D to 0xF63+0xFA6
    if debug and new_char != char:
        print(f'normalize char:{char} to {new_char}')

    return new_char

if __name__ == '__main__':
    text = 'ཞི་བའོ།      །འཆིང་བ་ཐམས་ཅད་ཀྱི་རྣམ་པར་གྲོལ་བའོ།       །ལྷག་མ་ཐམས་ཅད་ཀྱི་ཐར་པའོ།     །སེམས་ཉེ་བར་འཕྲོ་བ་ཐམས་ཅད་ཀྱི་ཞི་བའོ།     །འབྱོར་པ་ཐམས་ཅད་ཀྱི་འབྱུང་གནས་སོ།     །རྒུད་'
    char_list = string2char_list(text)
    print('end')



