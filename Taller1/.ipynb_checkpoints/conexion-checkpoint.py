import string
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import io
import os
from dotenv import load_dotenv

load_dotenv()


# se define la conexion y la dirección de la base de datos que se desea acceder
engine = create_engine('postgresql+psycopg2://root:root@postgreServer:5432/airQuality')


def new_model(df, name_model) -> None:
    """
    funcion encargada de crear un nuevo modelo en la base de datos 
    a partir de un cliente.
    df: Dataframe que contiene los datos que se llevara a la base de datos.
    name_model: string que contiene el nombre de la tabla que se va a crear.
    """

    df = df.rename_axis('id').reset_index()
    df['id'] = df.index+1
    df.head(0).to_sql(name_model, engine, if_exists='replace', index=False)


    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, name_model, null="")
    conn.commit()