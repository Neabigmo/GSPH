import sys as _sys
from pathlib import Path as _P
_F = _P(__file__).resolve()
_R = _F.parent
while _R != _R.parent and not ((_R / "src").is_dir() and (_R / "experiments").is_dir()):
    _R = _R.parent
for _d in (_R / "src", _R / "tasks" / "rna", _R / "tasks" / "protein_dna",
           _R / "tasks" / "membrane"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
for _e in sorted((_R / "experiments").glob("exp*")):
    if _e.is_dir() and str(_e) not in _sys.path:
        _sys.path.insert(0, str(_e))
p = '/work/home/acn8s56xc0/gsph_baselines/src/DFIRE-RNA/dfire/src/main.cc'
raw = open(p, 'rb').read().decode('utf-8', errors='replace')
lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')
NL = chr(92) + 'n'
final = []
for ln in lines:
    s = ln.strip()
    bad = ('sprintf' in s and 'sout' in s and
           ('%8.6f %8.6f' in s or 'sout);' in s or 'strlen' in s))
    bad = bad or s.startswith('strcat(sout') or s in (
        '", energy_norm_by_res, energy_norm_by_pair);', '");')
    if bad:
        if 'line.c_str()' in s:
            final.append('        sprintf (sout, "%s %8.6f ", line.c_str(), energy);')
        continue
    final.append(ln)

out = []
seen_norm = False
patched_else = False
for ln in final:
    out.append(ln)
    s = ln.strip()
    if s == 'double energy_norm_by_pair = 1.0 * energy / astr->n_atompairs;':
        seen_norm = True
        out.append('')
        out.append('            sprintf(sout + strlen(sout), "%8.6f %8.6f ' + NL + '", energy_norm_by_res, energy_norm_by_pair);')
    elif s == 'else {' and seen_norm and not patched_else:
        patched_else = True
        out.append('            strcat(sout, "' + NL + '");')
open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print('patched, norm=%s else=%s' % (seen_norm, patched_else))
