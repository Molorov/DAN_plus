
#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in Python  whose purpose is to
#  provide public implementation of deep learning works, in pytorch.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import re
import unicodedata as ucd

supported_lan = ['latin', # English, French, German, etc
                 'bo', # Tibetans
                 'cjk', # Chinese ideograph
                 'misc' # general purpose symbols
                 ]


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_sorted_lan_dict(text, main_lan=None):
    lan_dict = {}
    for chr in text:
        try:
            name = ucd.name(chr)
        except:
            continue
        if 'tibetan' in name.lower():
            lan = 'bo'
        elif 'latin' in name.lower():
            lan = 'latin'
        elif 'cjk' in name.lower() or chr in ['，', '。', '：']:
            lan = 'cjk'
        else:
            lan = 'misc'
        if lan not in lan_dict.keys():
            lan_dict[lan] = ''
        lan_dict[lan] += chr
    lan_dict = dict(sorted(lan_dict.items(), key=lambda item: len(item[1]), reverse=True))
    if 'misc' in lan_dict:
        misc = lan_dict['misc']
        del lan_dict['misc']
        if len(lan_dict):
            for lan in lan_dict:
                lan_dict[lan] += misc
        else:
            lan_dict[main_lan] = misc
    return lan_dict


