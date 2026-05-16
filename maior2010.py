import pandas as pd

INPUT_CSV = "matches.csv"
OUTPUT_CSV = "matches_2010_plus.csv"

# leitura tolerante a erros
df = pd.read_csv(
    INPUT_CSV,
    engine="python",
    on_bad_lines="skip"
)

# converte datas
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# filtra ano >= 2010
df_filtrado = df[df["date"].dt.year >= 2010]

# salva
df_filtrado.to_csv(OUTPUT_CSV, index=False)

print(f"Arquivo criado: {OUTPUT_CSV}")
print(f"Registros filtrados: {len(df_filtrado)}")

print(f"Linhas quebradas:")
with open("matches.csv", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        if i == 4899:
            print(line)
            break