'''INSTRUCCIONES
1. Carga y exploración de datos:
• Carga el dataset proporcionado en Spark.
• Convierte los datos en un RDD y un DataFrame.
• Explora los datos: muestra las primeras filas, el esquema y genera estadísticas descriptivas.

2. Procesamiento de datos con RDDs y DataFrames:
• Aplica transformaciones sobre los RDDs (filter, map, flatMap).
• Aplica acciones sobre los RDDs (collect, take, count).
• Realiza operaciones con DataFrames: filtrado, agregaciones y ordenamiento.
• Escribe los resultados en formato Parquet.'''



import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, lit
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import trim
import sys
import os



# Crear SparkSession
os.environ["PYSPARK_PYTHON"] = sys.executable
spark = SparkSession.builder.appName("AnalisisMigraciones").getOrCreate()

sc=spark.sparkContext

#Cargar archivo CSV

file_path='migraciones.csv'

#RDD
rdd = sc.textFile(file_path)

# Eliminar encabezado y convertir a tuplas
header = rdd.first()
rdd = rdd.filter(lambda x: x != header)
def parse_row(row):
    ID,Origen,Destino,Año,Razón,PIB_Origen,PIB_Destino,Tasa_Desempleo_Origen,Tasa_Desempleo_Destino,Nivel_Educativo_Origen,Nivel_Educativo_Destino,Población_Origen,Población_Destino = row.split(",")
    return (ID, Origen, Destino, int(Año),
            Razón, int(PIB_Origen), int(PIB_Destino),
            float(Tasa_Desempleo_Origen), float(Tasa_Desempleo_Destino),
            float(Nivel_Educativo_Origen), float(Nivel_Educativo_Destino),
            int(Población_Origen), int(Población_Destino)
    )
rdd_mapeado= rdd.map(parse_row)


rdd_mapeado.collect()
rdd_mapeado.take(1)
rdd_mapeado.count()

df_desde_rdd=rdd_mapeado.toDF(['ID','Origen','Destino','Año','Razón','PIB_Origen','PIB_Destino','Tasa_Desempleo_Origen','Tasa_Desempleo_Destino','Nivel_Educativo_Origen','Nivel_Educativo_Destino','Población_Origen','Población_Destino'])
df_desde_rdd.show()
df_desde_rdd.write.mode('overwrite').parquet('df_desde_rdd.parquet')


#DataFrame
df_raw = spark.read.csv(file_path, header=True, inferSchema=True)
df = df_raw

print(df.head())
print(df.describe().show())

#Exploración inicial
df.show(10)
df.printSchema()
for c in df.columns:
    nulls = df.filter(col(c).isNull()).count()
    print(f"Columna {c}: {nulls} valores nulos")

df.write.mode('overwrite').parquet('df_raw.parquet')


spark.stop()


'''3. Consultas con Spark SQL:
• Registra el DataFrame como una tabla temporal.
• Realiza consultas sobre los principales países de origen y destino.
• Analiza las principales razones de migración por región.'''

spark= SparkSession.builder.appName("AnalisisMigraciones").getOrCreate() # Restart spark session

df_parquet=spark.read.parquet('df_raw.parquet') # Use the df_raw.parquet generated above

df_parquet.show()
df_parquet.show(5, truncate=False)
df_parquet.printSchema()
df_parquet.count()
df_parquet.describe().show()

df_parquet.createOrReplaceTempView('migraciones')
resultado=spark.sql('SELECT Origen FROM migraciones WHERE PIB_Origen > 4000 ')
resultado.show()

resultado2=spark.sql('SELECT Destino FROM migraciones WHERE PIB_origen > 40000')
resultado2.show()

# Corrected SQL query for creating a new column 'Región'
new_df=spark.sql("""
    SELECT *,
        CASE
            WHEN Origen == 'Siria' THEN 'Medio Oriente'
            WHEN Origen == 'India' THEN 'Asia'
            ELSE 'América Latina'
        END AS Region
    FROM migraciones
""")

new_df.show()

# Corrected SQL query to analyze migration reasons by region
resultado3 = new_df.groupBy("Region", "Razón").count().orderBy("Region", col("count").desc())
resultado3.show()


'''4. Aplicación de MLlib para predicción de flujos migratorios:
• Convierte los datos en un formato adecuado para MLlib.
• Aplica un modelo de regresión logística para predecir la probabilidad de migración basada
en factores socioeconómicos.
• Evalúa el modelo y analiza su precisión.'''

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import lit


spark= SparkSession.builder.appName('Mlib_Supervised').getOrCreate()

data=spark.read.csv('migraciones.csv', header=True, inferSchema=True)

# Add a label column with value 1 for all rows
data = data.withColumn("label", lit(1))


# Handle categorical features
indexer = StringIndexer(inputCol="Razón", outputCol="reasonIndex")
data = indexer.fit(data).transform(data)
encoder = OneHotEncoder(inputCol="reasonIndex", outputCol="reasonVector")
data = encoder.fit(data).transform(data)


assembler=VectorAssembler(
    inputCols=['reasonVector', 'PIB_Origen', 'PIB_Destino', 'Tasa_Desempleo_Origen',
'Tasa_Desempleo_Destino', 'Nivel_Educativo_Origen', 'Nivel_Educativo_Destino',
'Población_Origen', 'Población_Destino'], outputCol='features')
data=assembler.transform(data)


# dividir en train y test

train, test=data.randomSplit([0.8, 0.2])

# crear modelo de regresión logística

lr=LogisticRegression(featuresCol='features', labelCol='label') # Replace 'label' with your actual label column name

# entrenar modelo
model=lr.fit(train)

# mostrar resultados
predictions = model.transform(test) # Transform the test data to get predictions

predictions.select('label', 'prediction').show() # Replace 'label' with your actual label column name

# evaluación con MAE
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_mae = RegressionEvaluator(
    labelCol="label", predictionCol="predictions", metricName="mae"
)
mae = evaluator_mae.evaluate(predictions)
print(f"Mean Absolute Error (MAE): {mae}")


spark.stop()

