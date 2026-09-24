from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
data=pd.read_csv(ROOT/'Base_de_datos.csv').head(2).drop(columns=['Pago_atiempo','puntaje'])
out=ROOT/'examples'; out.mkdir(exist_ok=True)
(out/'request.json').write_text(json.dumps({'records':json.loads(data.to_json(orient='records'))}, indent=2), encoding='utf-8')
