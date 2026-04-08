# HFI (Hedge Fund Inshallah) - Crypto Trading Bot

## Proje Ozeti
Bybit Futures uzerinde 3 motorlu (Trend Follower, Mean Reversion, Momentum Scalper) otonom kripto trading bot.
$100 baslangic, aylik $100 ekleme, kademeli leverage (3x->10x), snowball stratejisi.

## Mimari
```
Data (CCXT) -> Features (pandas-ta) -> Regime (ATR+ADX) -> Engines -> Risk -> Sizing -> Execution
```

## Kurallar

### Guvenlik
- API key'ler SADECE .env'de. ASLA hardcode etme.
- .env dosyasini ASLA commit etme.
- Live trading icin `--live` flag ZORUNLU. Default = paper trade.

### Kod Standartlari
- Python 3.12, type hints zorunlu.
- Pydantic v2 modeller: frozen=True, strict=True, extra='forbid'.
- Tum float alanlarinda NaN rejection (HFIModel base class).
- Finansal hesaplamalarda Decimal kullan (float DEGIL) - ozellikle order size ve fiyat.
- asyncio kullan I/O islemleri icin (CCXT async).
- Her engine YAML config'den parametre alir, hardcode YASAK.

### Risk Yonetimi
- Risk parametrelerini SESSIZCE degistirme. Her degisiklik loglanmali.
- Circuit breaker'i BYPASS etme.
- Max drawdown %15'i gecerse HALT. Manuel reset gerekli.
- Leverage milestone'lari config'de tanimli, kod icinde hardcode DEGIL.

### Test
- Her engine icin unit test zorunlu.
- Backtest sonuclari Sharpe > 1.0, Max DD < 15%, PF > 1.5 olmali.
- Paper trade en az 1 hafta basarili olmadan live'a gecme.

### Iletisim
- Turkce veya mixed TR/EN cevap ver.
- Finans kavramlarini basit acikla (kullanici finans bilmiyor).
