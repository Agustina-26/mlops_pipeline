from pathlib import Path
import nbformat
for path in (Path(__file__).resolve().parent.parent/'src').glob('*.ipynb'):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    if not nb.cells:
        raise ValueError(f'Notebook vacío: {path}')
    errors = [o for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error']
    if errors:
        raise ValueError(f'Notebook con errores guardados: {path}')
    print('OK:', path.name)
