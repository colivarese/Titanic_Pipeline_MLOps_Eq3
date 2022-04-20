from sre_constants import CATEGORY_UNI_LINEBREAK
import pandas as pd
import pytest


def obtener_datos_one_hot_encoder(df):
    titles = ["Mrs", "Mr", "Miss", "Master", "Other"]
    titles_len = len(titles)
    variables = ["title"]
    df = pd.DataFrame(titles, columns=["title"])
    return [titles_len, df, variables]


@pytest.mark.parametrize("result, df, variables", obtener_datos_one_hot_encoder())
def test_one_hot_encoding(len_one_hot, df, variables):
    # traer el OneHotEncoding de Cesar y hacer las pruebas, debe pasar de 1 columna a 5
    assert df.shape[1] == len_one_hot
