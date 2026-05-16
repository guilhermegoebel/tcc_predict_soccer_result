import csv
import json
import os
import re
import time
from datetime import datetime
import pandas as pd
import random

import requests
from bs4 import BeautifulSoup

OUTPUT_CSV = "transfermarkt_valores.csv"
MATCHES_CSV = "teste.csv"

SLEEP_ENTRE = 10
MAX_RETRIES = 3

HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
}

HEADERS_API = {
    **HEADERS_HTML,
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS_HTML)

FIELDNAMES = [
    "player_search_name",
    "tm_id",
    "tm_name",
    "tm_url",
    "date",
    "date_raw",
    "market_value_eur",
    "market_value_str",
    "club",
]

def carregar_jogadores(csv_path):
    df = pd.read_csv(csv_path)

    jogadores = []

    for players_str in df["home_players"].dropna():
        lista = players_str.split("|")
        ultimos_5 = lista[-5:]

        jogadores.extend(ultimos_5)

    # remove duplicados
    jogadores_unicos = list(dict.fromkeys(jogadores))

    return jogadores_unicos

PLAYERS = carregar_jogadores(MATCHES_CSV)


def get_with_retry(url, headers=None, params=None, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            r = SESSION.get(
                url,
                headers=headers or HEADERS_HTML,
                params=params,
                timeout=15,
            )
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"      [RATE LIMIT] Aguardando {wait}s...")
                time.sleep(wait)
            else:
                print(f"      [HTTP {r.status_code}] {url[:60]}")
                time.sleep(3)
        except requests.RequestException as e:
            print(f"      [ERRO] {e} (tentativa {attempt + 1}/{retries})")
            time.sleep(3 * (attempt + 1))
    return None


def buscar_tm_id_e_url(nome_jogador):
    url_busca = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    params = {"query": nome_jogador, "x": "0", "y": "0"}

    print(f"    Buscando '{nome_jogador}' no Transfermarkt...", end="  ")
    r = get_with_retry(url_busca, params=params)
    if not r:
        print("falhou")
        return None, None, None

    soup = BeautifulSoup(r.text, "html.parser")
    resultados = []

    for a in soup.select("table.items td.hauptlink a[href*='/profil/spieler/']"):
        href = a.get("href", "")
        nome = a.text.strip()
        m = re.search(r"/spieler/(\d+)", href)
        if m and nome:
            tm_id = m.group(1)
            tm_url = f"https://www.transfermarkt.com{href}"
            resultados.append((tm_id, tm_url, nome))

    if not resultados:
        for a in soup.find_all("a", href=re.compile(r"/spieler/\d+")):
            href = a.get("href", "")
            nome = a.text.strip()
            m = re.search(r"/spieler/(\d+)", href)
            if m and nome and len(nome) > 2:
                tm_id = m.group(1)
                tm_url = f"https://www.transfermarkt.com{href}"
                resultados.append((tm_id, tm_url, nome))
                if len(resultados) >= 5:
                    break

    if not resultados:
        print("nao encontrado")
        return None, None, None

    tm_id, tm_url, nome_tm = resultados[0]
    print(f"encontrado: {nome_tm} (ID: {tm_id})")
    return tm_id, tm_url, nome_tm


def buscar_tm_id_via_perfil(nome_jogador):
    slug = nome_jogador.lower()
    slug = re.sub(r"['\.]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    url = f"https://www.transfermarkt.com/{slug}/profil/spieler/1"
    r = get_with_retry(url)
    if r:
        m = re.search(r"/spieler/(\d+)", r.url)
        if m:
            return m.group(1), r.url
    return None, None


def normalizar_data(date_str):
    if not date_str:
        return ""
    formatos = [
        "%b %d, %Y",
        "%B %d, %Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%m/%d/%Y",
    ]
    for fmt in formatos:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def obter_historico_valores(tm_id):
    url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{tm_id}"
    r = get_with_retry(url, headers=HEADERS_API)
    if not r:
        return []

    try:
        data = r.json()
    except Exception:
        print(f"      [ERRO JSON] Resposta nao-JSON para player {tm_id}")
        return []

    registros = []
    mv_list = data.get("list") or data.get("marketValues") or []

    for item in mv_list:
        date_raw = item.get("datum_mw") or item.get("date") or item.get("datetime") or ""
        value_raw = item.get("mwValue") or item.get("value") or 0
        value_str = item.get("mw") or item.get("marketValue") or ""
        club = item.get("verein") or item.get("club") or item.get("clubName") or ""
        registros.append(
            {
                "date_raw": date_raw,
                "date": normalizar_data(date_raw),
                "value_eur": int(value_raw) if value_raw else 0,
                "value_str": value_str,
                "club": club,
            }
        )

    return registros


def obter_historico_valores_html(tm_id, tm_url):
    url = re.sub(r"/profil/", "/marktwertverlauf/", tm_url)
    if "/marktwertverlauf/" not in url:
        url = f"https://www.transfermarkt.com/x/marktwertverlauf/spieler/{tm_id}"

    r = get_with_retry(url)
    if not r:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    registros = []

    for script in soup.find_all("script"):
        src = script.string or ""
        if "marketValues" in src or "mwValue" in src or "datum_mw" in src:
            m = re.search(r'"list"\s*:\s*(\[.*?\])', src, re.DOTALL)
            if not m:
                m = re.search(r'marketValues\s*=\s*(\[.*?\])', src, re.DOTALL)
            if m:
                try:
                    lista = json.loads(m.group(1))
                    for item in lista:
                        date_raw = item.get("datum_mw") or item.get("date") or ""
                        value_raw = item.get("mwValue") or item.get("value") or 0
                        value_str = item.get("mw") or ""
                        club = item.get("verein") or item.get("club") or ""
                        registros.append(
                            {
                                "date_raw": date_raw,
                                "date": normalizar_data(date_raw),
                                "value_eur": int(value_raw) if value_raw else 0,
                                "value_str": value_str,
                                "club": club,
                            }
                        )
                    if registros:
                        return registros
                except json.JSONDecodeError:
                    pass

    return registros


def salvar_registros(registros_csv):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(registros_csv)


def carregar_progresso():
    feitos = set()
    if not os.path.exists(OUTPUT_CSV):
        return feitos, []
    linhas = []
    try:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                linhas.append(row)
                feitos.add(row.get("player_search_name", ""))
    except Exception:
        pass
    return feitos, linhas


def main():
    print("=" * 60)
    print("  Transfermarkt Market Value History Scraper")
    print("=" * 60)

    inicio = datetime.now()
    ja_feitos, todas_linhas = carregar_progresso()

    if ja_feitos:
        print(f"\n  Retomando: {len(set(r['player_search_name'] for r in todas_linhas))} jogadores já no CSV")

    total_valores = 0
    jogadores_ok = 0
    jogadores_nao_encontrados = []

    for i, nome in enumerate(PLAYERS, 1):
        chave = nome

        print(f"\n[{i:>3}/{len(PLAYERS)}] {nome}")

        if chave in ja_feitos:
            n = sum(1 for r in todas_linhas if r["player_search_name"] == nome)
            print(f"        Ja coletado ({n} registros de valor)")
            continue

        tm_id, tm_url, tm_nome = buscar_tm_id_e_url(nome)

        if not tm_id:
            nome_simples = re.sub(
                r"\b(de|van|den|der|von|do|da|di|du|le|la)\b",
                "",
                nome,
                flags=re.I,
            ).strip()
            nome_simples = re.sub(r"\s+", " ", nome_simples)
            if nome_simples != nome:
                print(f"    Tentando nome simplificado: '{nome_simples}'")
                tm_id, tm_url, tm_nome = buscar_tm_id_e_url(nome_simples)

        if not tm_id:
            partes = nome.split()
            if len(partes) > 1:
                sobrenome = partes[-1]
                print(f"    Tentando só sobrenome: '{sobrenome}'")
                tm_id, tm_url, tm_nome = buscar_tm_id_e_url(sobrenome)

        if not tm_id:
            slug = re.sub(r"['\.]", "", nome.lower())
            slug = re.sub(r"\s+", "-", slug.strip())
            slug = re.sub(r"[^a-z0-9\-]", "", slug)
            url = f"https://www.transfermarkt.com/{slug}/profil/spieler/1"
            print(f"    Tentando perfil direto: '{url}'")
            r = get_with_retry(url)
            if r:
                m = re.search(r"/spieler/(\d+)", r.url)
                if m:
                    tm_id = m.group(1)
                    tm_url = r.url
                    tm_nome = nome

        if not tm_id:
            print("    !! NAO ENCONTRADO no Transfermarkt")
            jogadores_nao_encontrados.append(nome)
            todas_linhas.append(
                {
                    "player_search_name": nome,
                    "tm_id": "NAO_ENCONTRADO",
                    "tm_name": "",
                    "tm_url": "",
                    "date": "",
                    "date_raw": "",
                    "market_value_eur": "",
                    "market_value_str": "",
                    "club": "",
                }
            )
            salvar_registros(todas_linhas)
            time.sleep(random.uniform(5, 10))
            continue

        print(f"        TM ID:  {tm_id}")
        print(f"        TM URL: {tm_url}")

        time.sleep(random.uniform(5, 10))

        print(f"    Buscando histórico de valores...", end="  ")
        historico = obter_historico_valores(tm_id)

        if not historico:
            print("endpoint vazio, tentando fallback HTML...")
            historico = obter_historico_valores_html(tm_id, tm_url)

        if not historico:
            print("nenhum histórico encontrado")
            todas_linhas.append(
                {
                    "player_search_name": nome,
                    "tm_id": tm_id,
                    "tm_name": tm_nome,
                    "tm_url": tm_url,
                    "date": "",
                    "date_raw": "",
                    "market_value_eur": 0,
                    "market_value_str": "sem dados",
                    "club": "",
                }
            )
        else:
            print(f"{len(historico)} snapshots de valor")
            for snapshot in historico:
                todas_linhas.append(
                    {
                        "player_search_name": nome,
                        "tm_id": tm_id,
                        "tm_name": tm_nome,
                        "tm_url": tm_url,
                        "date": snapshot["date"],
                        "date_raw": snapshot["date_raw"],
                        "market_value_eur": snapshot["value_eur"],
                        "market_value_str": snapshot["value_str"],
                        "club": snapshot["club"],
                    }
                )
            total_valores += len(historico)
            jogadores_ok += 1

        salvar_registros(todas_linhas)
        time.sleep(random.uniform(5, 10))

    dur = datetime.now() - inicio
    print("\n" + "=" * 60)
    print(f"  Concluido em {dur}")
    print(f"  Jogadores com histórico: {jogadores_ok}/{len(PLAYERS)}")
    print(f"  Total de snapshots:      {total_valores:,}")
    print(f"  Arquivo: {os.path.abspath(OUTPUT_CSV)}")

    if jogadores_nao_encontrados:
        print(f"\n  NAO ENCONTRADOS ({len(jogadores_nao_encontrados)}):")
        for nome in jogadores_nao_encontrados:
            print(f"    - {nome}")
    print("=" * 60)


if __name__ == "__main__":
    main()