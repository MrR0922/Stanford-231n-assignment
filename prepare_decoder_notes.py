from pathlib import Path
p = Path(r'C:\Obsidian\脑机接口\2026.10.10 Transformer.md')
s = p.read_text(encoding='utf-8-sig')
a = s.index('## 8. Transformer Decoder 与图像描述')
b = s.index('## 9. Vision Transformer', a)
section = Path('decoder_section.md').read_text(encoding='utf-8-sig')
Path('transformer_notes_updated.md').write_text(s[:a]+section+'\n'+s[b:], encoding='utf-8')
assert s[:a] in Path('transformer_notes_updated.md').read_text(encoding='utf-8')
print('Updated section 8; preserved other sections.')
