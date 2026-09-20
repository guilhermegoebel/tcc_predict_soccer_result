import csv
import json
import os
import re
import time
import random
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    import cloudscraper
    def _make_session():
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
except ImportError:
    import requests
    def _make_session():
        return requests.Session()

PLAYERS_CSV   = "players.csv"
OUTPUT_CSV    = "transfermarkt_valores.csv"
DELAY_MIN     = 3.0
DELAY_MAX     = 7.0
DELAY_API_MIN = 1.5
DELAY_API_MAX = 3.5
MAX_RETRIES   = 4

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


def make_session():
    s = _make_session()
    s.headers.update(HEADERS_HTML)
    return s


SESSION = make_session()


def jitter_sleep(min_s, max_s):
    time.sleep(random.uniform(min_s, max_s))


def clean_name(name):
    name = name.strip()
    name = re.sub(r"^['\"`\u2018\u2019\u201c\u201d\s]+", "", name)
    name = re.sub(r"['\"`\u2018\u2019\u201c\u201d\s]+$", "", name)
    return name.strip()


_CHAR_MAP = str.maketrans({
    "ı": "i",  "İ": "I",
    "ə": "e",  "Ə": "E",
    "ğ": "g",  "Ğ": "G",
    "ß": "ss",
    "æ": "ae", "Æ": "AE",
    "ø": "o",  "Ø": "O",
    "þ": "th", "Þ": "TH",
    "ð": "d",  "Ð": "D",
    "ł": "l",  "Ł": "L",
    "đ": "d",  "Đ": "D",
    "ħ": "h",  "Ħ": "H",
    "ŋ": "n",  "Ŋ": "N",
    "œ": "oe", "Œ": "OE",
})

def ascii_fallback(name):
    name = name.translate(_CHAR_MAP)
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def expand_initials(name):
    expanded = re.sub(r"\b([A-Za-z])\.(?:\s*([A-Za-z])\.)*", lambda m: m.group(0).replace(".", "").replace(" ", ""), name)
    expanded = re.sub(r"\s+", " ", expanded).strip()
    return expanded if expanded != name else None


def get_with_retry(url, headers=None, params=None, retries=MAX_RETRIES):
    global SESSION
    for attempt in range(retries):
        try:
            r = SESSION.get(
                url,
                headers=headers or HEADERS_HTML,
                params=params,
                timeout=20,
            )
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 15 * (attempt + 1)))
                print(f"      [429] Aguardando {retry_after}s...")
                time.sleep(retry_after)
                SESSION = make_session()
                continue
            if r.status_code in (403, 503):
                wait = 20 * (attempt + 1)
                print(f"      [HTTP {r.status_code}] Aguardando {wait}s e recriando sessão...")
                time.sleep(wait)
                SESSION = make_session()
                continue
            print(f"      [HTTP {r.status_code}] {url[:70]}")
            time.sleep(5)
        except Exception as exc:
            wait = 5 * (attempt + 1)
            print(f"      [ERRO] {exc} (tentativa {attempt + 1}/{retries}) aguardando {wait}s")
            time.sleep(wait)
    return None


def buscar_tm(query):
    from bs4 import BeautifulSoup
    url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    r = get_with_retry(url, params={"query": query, "x": "0", "y": "0"})
    if not r:
        return None, None, None

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.select("table.items td.hauptlink a[href*='/profil/spieler/']"):
        href = a.get("href", "")
        nome_tm = a.text.strip()
        m = re.search(r"/spieler/(\d+)", href)
        if m and nome_tm:
            return m.group(1), f"https://www.transfermarkt.com{href}", nome_tm

    for a in soup.find_all("a", href=re.compile(r"/spieler/\d+")):
        href = a.get("href", "")
        nome_tm = a.text.strip()
        m = re.search(r"/spieler/(\d+)", href)
        if m and len(nome_tm) > 2:
            return m.group(1), f"https://www.transfermarkt.com{href}", nome_tm

    return None, None, None


def buscar_com_fallbacks(nome_original):
    nome = clean_name(nome_original)
    prefixos = {"de", "van", "den", "der", "von", "do", "da", "di", "du", "le", "la", "el"}

    tentativas = []

    tentativas.append(("original", nome))

    nome_ascii = ascii_fallback(nome)
    if nome_ascii != nome:
        tentativas.append(("ascii", nome_ascii))

    nome_initials = expand_initials(nome)
    if nome_initials and nome_initials != nome:
        tentativas.append(("iniciais expandidas", nome_initials))
        nome_initials_ascii = ascii_fallback(nome_initials)
        if nome_initials_ascii != nome_initials:
            tentativas.append(("iniciais+ascii", nome_initials_ascii))

    palavras = nome.split()
    palavras_sem_prefixo = [p for p in palavras if p.lower() not in prefixos]
    nome_sem_prefixo = " ".join(palavras_sem_prefixo)
    if nome_sem_prefixo != nome and len(nome_sem_prefixo) > 2:
        tentativas.append(("sem prefixo", nome_sem_prefixo))
        nome_sp_ascii = ascii_fallback(nome_sem_prefixo)
        if nome_sp_ascii != nome_sem_prefixo:
            tentativas.append(("sem prefixo+ascii", nome_sp_ascii))

    seen = set()
    for label, query in tentativas:
        if query in seen:
            continue
        seen.add(query)
        print(f"    [{label}] Buscando '{query}'...", end="  ", flush=True)
        tm_id, tm_url, tm_nome = buscar_tm(query)
        if tm_id:
            print(f"encontrado: {tm_nome} (ID: {tm_id})")
            return tm_id, tm_url, tm_nome
        print("não encontrado")
        jitter_sleep(1.0, 2.5)

    return None, None, None


def normalizar_data(date_str):
    if not date_str:
        return ""
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def obter_historico_api(tm_id):
    url = f"https://www.transfermarkt.com/ceapi/marketValueDevelopment/graph/{tm_id}"
    r = get_with_retry(url, headers=HEADERS_API)
    if not r:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    registros = []
    for item in (data.get("list") or data.get("marketValues") or []):
        date_raw  = item.get("datum_mw") or item.get("date") or ""
        value_raw = item.get("mwValue") or item.get("value") or 0
        value_str = item.get("mw") or item.get("marketValue") or ""
        club      = item.get("verein") or item.get("club") or ""
        registros.append({
            "date_raw":  date_raw,
            "date":      normalizar_data(date_raw),
            "value_eur": int(value_raw) if value_raw else 0,
            "value_str": value_str,
            "club":      club,
        })
    return registros


def obter_historico_html(tm_id, tm_url):
    from bs4 import BeautifulSoup
    url = re.sub(r"/profil/", "/marktwertverlauf/", tm_url)
    if "/marktwertverlauf/" not in url:
        url = f"https://www.transfermarkt.com/x/marktwertverlauf/spieler/{tm_id}"
    r = get_with_retry(url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    for script in soup.find_all("script"):
        src = script.string or ""
        if "marketValues" not in src and "mwValue" not in src and "datum_mw" not in src:
            continue
        m = re.search(r'"list"\s*:\s*(\[.*?\])', src, re.DOTALL)
        if not m:
            m = re.search(r'marketValues\s*=\s*(\[.*?\])', src, re.DOTALL)
        if m:
            try:
                lista = json.loads(m.group(1))
                registros = []
                for item in lista:
                    date_raw  = item.get("datum_mw") or item.get("date") or ""
                    value_raw = item.get("mwValue") or item.get("value") or 0
                    value_str = item.get("mw") or ""
                    club      = item.get("verein") or item.get("club") or ""
                    registros.append({
                        "date_raw":  date_raw,
                        "date":      normalizar_data(date_raw),
                        "value_eur": int(value_raw) if value_raw else 0,
                        "value_str": value_str,
                        "club":      club,
                    })
                if registros:
                    return registros
            except json.JSONDecodeError:
                pass
    return []


def carregar_ja_feitos():
    feitos = set()
    linhas = []
    if not Path(OUTPUT_CSV).exists():
        return feitos, linhas
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            linhas.append(row)
            nome = row.get("player_search_name", "").strip()
            if nome:
                feitos.add(nome)
    return feitos, linhas


def salvar(linhas):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(linhas)


def ler_players_csv():
    players = []
    with open(PLAYERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nome = row.get("player", "").strip()
            if nome:
                players.append(nome)
    return players


def main():
    print("=" * 60)
    print("  Transfermarkt Market Value Scraper")
    print(f"  Lendo jogadores de: {PLAYERS_CSV}")
    print("=" * 60)

    players = ler_players_csv()
    ja_feitos, todas_linhas = carregar_ja_feitos()

    pendentes = [p for p in players if clean_name(p) not in ja_feitos and p not in ja_feitos]

    print(f"\n  Total no players.csv:  {len(players)}")
    print(f"  Já processados:        {len(ja_feitos)}")
    print(f"  Pendentes:             {len(pendentes)}\n")

    nao_encontrados = []
    snapshots_total = 0

    for i, nome_original in enumerate(pendentes, 1):
        nome_chave = clean_name(nome_original)
        print(f"\n[{i:>5}/{len(pendentes)}] {nome_original}")

        tm_id, tm_url, tm_nome = buscar_com_fallbacks(nome_original)

        if not tm_id:
            nao_encontrados.append(nome_original)
            todas_linhas.append({
                "player_search_name": nome_chave,
                "tm_id":             "NAO_ENCONTRADO",
                "tm_name":           "",
                "tm_url":            "",
                "date":              "",
                "date_raw":          "",
                "market_value_eur":  "",
                "market_value_str":  "",
                "club":              "",
            })
            salvar(todas_linhas)
            jitter_sleep(DELAY_MIN, DELAY_MAX)
            continue

        jitter_sleep(DELAY_API_MIN, DELAY_API_MAX)

        print(f"    Buscando histórico...", end="  ", flush=True)
        historico = obter_historico_api(tm_id)
        if not historico:
            print("endpoint vazio, tentando HTML...", end="  ", flush=True)
            jitter_sleep(DELAY_API_MIN, DELAY_API_MAX)
            historico = obter_historico_html(tm_id, tm_url)

        if not historico:
            print("sem histórico")
            todas_linhas.append({
                "player_search_name": nome_chave,
                "tm_id":             tm_id,
                "tm_name":           tm_nome,
                "tm_url":            tm_url,
                "date":              "",
                "date_raw":          "",
                "market_value_eur":  0,
                "market_value_str":  "sem dados",
                "club":              "",
            })
        else:
            print(f"{len(historico)} snapshots")
            snapshots_total += len(historico)
            for snap in historico:
                todas_linhas.append({
                    "player_search_name": nome_chave,
                    "tm_id":             tm_id,
                    "tm_name":           tm_nome,
                    "tm_url":            tm_url,
                    "date":              snap["date"],
                    "date_raw":          snap["date_raw"],
                    "market_value_eur":  snap["value_eur"],
                    "market_value_str":  snap["value_str"],
                    "club":              snap["club"],
                })

        salvar(todas_linhas)
        jitter_sleep(DELAY_MIN, DELAY_MAX)

    print("\n" + "=" * 60)
    print(f"  Concluído")
    print(f"  Snapshots coletados:  {snapshots_total:,}")
    print(f"  Não encontrados:      {len(nao_encontrados)}")
    if nao_encontrados:
        for n in nao_encontrados[:20]:
            print(f"    - {n}")
        if len(nao_encontrados) > 20:
            print(f"    ... e mais {len(nao_encontrados) - 20}")
    print("=" * 60)


if __name__ == "__main__":
    main()