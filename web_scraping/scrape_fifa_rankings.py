import csv, os, time, re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, StaleElementReferenceException, ElementClickInterceptedException
)

# ── CONFIGURACOES ───────────────────────────────────────
BRAVE_PATH  = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
BASE_URL    = "https://inside.fifa.com/fifa-world-ranking/men"
OUTPUT_CSV  = "fifa_rankings_historico.csv"
ANOS_ALVO   = list(range(1992, 2026))
HEADLESS    = False
SLEEP_TABELA = 3.0
# ───────────────────────────────────────────────────────

SEL_RANK        = ".custom-rank-cell_rankNumber__RORLl"
SEL_TEAM        = ".custom-team-cell_teamName__c_tEs"
SEL_POINTS      = ".custom-points-cell_points__Lt6_7 span"
SEL_TABLE_BODY  = "tbody tr"
SEL_SHOW_ALL    = "button[aria-label='Show full rankings']"
SEL_OPEN_FILTERS= ".live-ranking-mobile-filter_newFilterButtonMobile__TudCs"
SEL_DRAWER_OPEN = ".drawer_drawerContent__ER2sc"
SEL_FILTER_NAME = ".radio-group-filter-module_filterName__TGB-4"
SEL_RADIO_ITEM  = ".radio-button-module_radio__hkrkE"
SEL_SHOW_MORE   = ".radio-group-filter-module_showMoreButton__rJgEB"
SEL_APPLY       = "button[aria-label='Apply']"


def criar_driver():
    opts = Options()
    opts.binary_location = BRAVE_PATH
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=390,844")   # mobile — drawer funciona aqui
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=opts)
    driver.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    return driver


def fechar_cookies(driver):
    for sel in ["#onetrust-reject-all-handler", "#onetrust-accept-btn-handler"]:
        try:
            btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            btn.click()
            time.sleep(1.5)
            return
        except Exception:
            continue
    try:
        driver.execute_script("""
            ['onetrust-consent-sdk','onetrust-overlay','onetrust-banner-sdk'].forEach(function(id){
                var el=document.getElementById(id); if(el) el.remove();
            });
            document.body.style.overflow='auto';
        """)
        time.sleep(0.5)
    except Exception:
        pass


def aguardar_tabela(driver):
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, SEL_RANK))
        )
        time.sleep(SLEEP_TABELA)
    except TimeoutException:
        pass


def abrir_drawer(driver):
    try:
        driver.execute_script(
            "var s=document.getElementById('onetrust-consent-sdk');if(s)s.remove();"
        )
    except Exception:
        pass
    for _ in range(3):
        try:
            btn = WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SEL_OPEN_FILTERS))
            )
            driver.execute_script("arguments[0].click();", btn)
            WebDriverWait(driver, 8).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, SEL_DRAWER_OPEN))
            )
            time.sleep(0.8)
            return True
        except Exception:
            time.sleep(1.0)
    return False


def fechar_drawer(driver):
    try:
        from selenium.webdriver.common.keys import Keys
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(0.6)
    except Exception:
        pass


def encontrar_secao(driver, nome):
    """Retorna o container pai do filtro com filterName == nome."""
    for el in driver.find_elements(By.CSS_SELECTOR, SEL_FILTER_NAME):
        try:
            if el.text.strip() == nome:
                return driver.execute_script("return arguments[0].parentElement;", el)
        except StaleElementReferenceException:
            continue
    return None


def expandir_todos_anos(driver, container):
    """
    Clica em 'Show more' repetidamente até sumir o botao,
    garantindo que TODOS os anos sejam carregados.
    """
    MAX_CLICKS = 10
    for _ in range(MAX_CLICKS):
        try:
            btn = container.find_element(By.CSS_SELECTOR, SEL_SHOW_MORE)
            if btn.is_displayed():
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.8)
            else:
                break
        except Exception:
            break   # botao sumiu — todos os anos foram carregados


def listar_anos_drawer(driver):
    """
    Abre o drawer, expande TODOS os anos (clicando Show more ate o fim),
    retorna lista de anos disponiveis como strings, fecha o drawer.
    """
    if not abrir_drawer(driver):
        return []

    container = encontrar_secao(driver, "Year")
    if not container:
        fechar_drawer(driver)
        return []

    expandir_todos_anos(driver, container)

    anos = []
    for item in container.find_elements(By.CSS_SELECTOR, SEL_RADIO_ITEM):
        try:
            lbl = item.find_element(By.TAG_NAME, "label")
            txt = lbl.text.strip()
            if re.match(r"^\d{4}$", txt):
                anos.append(txt)
        except Exception:
            continue

    fechar_drawer(driver)
    return sorted(anos, key=int)


def listar_datas_para_ano(driver, ano):
    """
    Abre drawer, seleciona o ano (sem Apply), coleta as datas, fecha.
    """
    if not abrir_drawer(driver):
        return []

    # Seleciona ano
    container_year = encontrar_secao(driver, "Year")
    if container_year:
        expandir_todos_anos(driver, container_year)
        for item in container_year.find_elements(By.CSS_SELECTOR, SEL_RADIO_ITEM):
            try:
                lbl = item.find_element(By.TAG_NAME, "label")
                if lbl.text.strip() == str(ano):
                    driver.execute_script("arguments[0].click();", lbl)
                    time.sleep(0.8)
                    break
            except Exception:
                continue

    # Coleta datas
    datas = []
    container_date = encontrar_secao(driver, "Date")
    if container_date:
        for item in container_date.find_elements(By.CSS_SELECTOR, SEL_RADIO_ITEM):
            try:
                lbl = item.find_element(By.TAG_NAME, "label")
                txt = lbl.text.strip()
                if txt:
                    datas.append(txt)
            except Exception:
                continue

    fechar_drawer(driver)
    return datas


def aplicar_ano_data(driver, ano, data):
    """Abre drawer, seleciona ano + data, clica Apply, aguarda tabela."""
    if not abrir_drawer(driver):
        return False

    # Ano
    container_year = encontrar_secao(driver, "Year")
    ok_ano = False
    if container_year:
        expandir_todos_anos(driver, container_year)
        for item in container_year.find_elements(By.CSS_SELECTOR, SEL_RADIO_ITEM):
            try:
                lbl = item.find_element(By.TAG_NAME, "label")
                if lbl.text.strip() == str(ano):
                    driver.execute_script("arguments[0].click();", lbl)
                    ok_ano = True
                    time.sleep(0.8)
                    break
            except Exception:
                continue

    # Data
    container_date = encontrar_secao(driver, "Date")
    ok_data = False
    if container_date:
        for item in container_date.find_elements(By.CSS_SELECTOR, SEL_RADIO_ITEM):
            try:
                lbl = item.find_element(By.TAG_NAME, "label")
                if lbl.text.strip() == data:
                    driver.execute_script("arguments[0].click();", lbl)
                    ok_data = True
                    time.sleep(0.4)
                    break
            except Exception:
                continue

    if not (ok_ano or ok_data):
        fechar_drawer(driver)
        return False

    # Apply
    try:
        btn = WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SEL_APPLY))
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(1.0)
        aguardar_tabela(driver)
        return True
    except TimeoutException:
        return False


def clicar_mostrar_todos(driver):
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, SEL_SHOW_ALL))
        )
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(2.5)
    except TimeoutException:
        pass


def extrair_tabela(driver, year, date_label):
    registros = []
    for linha in driver.find_elements(By.CSS_SELECTOR, SEL_TABLE_BODY):
        try:
            r = linha.find_elements(By.CSS_SELECTOR, SEL_RANK)
            t = linha.find_elements(By.CSS_SELECTOR, SEL_TEAM)
            p = linha.find_elements(By.CSS_SELECTOR, SEL_POINTS)
            rank   = r[0].text.strip() if r else ""
            team   = t[0].text.strip() if t else ""
            points = p[0].text.strip() if p else ""
            if rank and team:
                registros.append({"year": year, "date_label": date_label,
                                   "rank": rank, "team": team, "points": points})
        except StaleElementReferenceException:
            continue
    if not registros:
        ranks = driver.find_elements(By.CSS_SELECTOR, SEL_RANK)
        teams = driver.find_elements(By.CSS_SELECTOR, SEL_TEAM)
        pts   = driver.find_elements(By.CSS_SELECTOR, SEL_POINTS)
        for i, (rk, tm) in enumerate(zip(ranks, teams)):
            try:
                pt = pts[i].text.strip() if i < len(pts) else ""
                registros.append({"year": year, "date_label": date_label,
                                   "rank": rk.text.strip(), "team": tm.text.strip(), "points": pt})
            except StaleElementReferenceException:
                continue
    return registros


def salvar_csv(registros):
    if not registros:
        return
    escrever_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "date_label", "rank", "team", "points"])
        if escrever_header:
            w.writeheader()
        w.writerows(registros)


def ja_coletados():
    feitos = set()
    if not os.path.exists(OUTPUT_CSV):
        return feitos
    try:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                feitos.add((row.get("year", ""), row.get("date_label", "")))
    except Exception:
        pass
    return feitos


def main():
    print("=" * 60)
    print("  FIFA Ranking — Complementar 1993 a 2012")
    print("=" * 60)

    inicio = datetime.now()
    total = 0
    coletados = ja_coletados()

    anos_ja_no_csv = set(y for y, _ in coletados)
    anos_faltando = [str(a) for a in ANOS_ALVO if str(a) not in anos_ja_no_csv]

    if not anos_faltando:
        print("\n  Todos os anos 1993-2012 ja estao no CSV. Nada a fazer.")
        return

    print(f"\n  Anos a coletar: {anos_faltando[0]} -> {anos_faltando[-1]}")
    print(f"  ({len(anos_faltando)} anos)")

    driver = criar_driver()
    try:
        print(f"\n  Abrindo: {BASE_URL}")
        driver.get(BASE_URL)
        fechar_cookies(driver)
        print("  Aguardando tabela...")
        aguardar_tabela(driver)
        time.sleep(2)

        # Verifica se os anos alvo existem no drawer
        print("\n  Verificando anos disponiveis no drawer...")
        anos_disponiveis = listar_anos_drawer(driver)
        print(f"  {len(anos_disponiveis)} anos encontrados: {anos_disponiveis[0]} -> {anos_disponiveis[-1]}")

        anos_para_processar = [a for a in anos_faltando if a in anos_disponiveis]
        if not anos_para_processar:
            print("  ERRO: Nenhum dos anos alvo foi encontrado no drawer.")
            return
        print(f"  Anos a processar: {', '.join(anos_para_processar)}\n")

        for i_ano, ano in enumerate(anos_para_processar):
            print(f"\n  -- {ano}  ({i_ano+1}/{len(anos_para_processar)}) --")

            datas = listar_datas_para_ano(driver, ano)
            if not datas:
                print(f"    [WARN] Sem datas para {ano}")
                continue
            print(f"    {len(datas)} datas: {', '.join(datas)}")

            for i_d, data in enumerate(datas):
                chave = (str(ano), data)
                if chave in coletados:
                    print(f"    [{i_d+1:>2}/{len(datas)}] {data:<22}  ja coletado")
                    continue

                print(f"    [{i_d+1:>2}/{len(datas)}] {data:<22}", end="  ")

                ok = aplicar_ano_data(driver, ano, data)
                if not ok:
                    print("[ERRO] filtro nao aplicado")
                    continue

                clicar_mostrar_todos(driver)
                registros = extrair_tabela(driver, str(ano), data)

                if registros:
                    salvar_csv(registros)
                    total += len(registros)
                    coletados.add(chave)
                    print(f"OK  {len(registros):>3} times")
                else:
                    print("[WARN] sem registros")

    except KeyboardInterrupt:
        print("\n[INFO] Interrompido. Progresso salvo.")
    finally:
        driver.quit()

    dur = datetime.now() - inicio
    print("\n" + "=" * 60)
    print(f"  Tempo: {dur}")
    print(f"  Registros adicionados: {total:,}")
    print(f"  Arquivo: {os.path.abspath(OUTPUT_CSV)}")
    print("=" * 60)


if __name__ == "__main__":
    main()