from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_loads_and_changes_variable():
    path = Path(__file__).resolve().parent.parent/'src/app.py'
    app = AppTest.from_file(str(path), default_timeout=45).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.selectbox[0].select('tipo_laboral').run()
    assert not app.exception
    app.slider[1].set_value(.30).run()
    assert not app.exception
