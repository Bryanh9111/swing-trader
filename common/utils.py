"""Shared utilities used across modules.

This module contains reusable helpers that are needed by both the Universe
Builder and Scanner components.
"""

from __future__ import annotations

import re

__all__ = ["is_etf"]


_ETF_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        # Broad market / style / size
        r"^SPY$",
        r"^QQQ$",
        r"^QQQ[A-Z]$",      # QQQ variants: QQQM, QQQI, QQQE, QQQJ
        r"^IWM$",
        r"^DIA$",
        r"^MDY$",
        r"^MDY[A-Z]$",      # MDY variants: MDYG, MDYV
        r"^IVV$",
        r"^VTI$",
        r"^VOO$",
        r"^VT$",
        r"^RSP$",
        r"^IBB$",           # Biotech ETF
        # Sector SPDR (XL*)
        r"^XL[A-Z]$",
        # Common issuer / regional families
        r"^DF[A-Z]{2,3}$",  # Dimensional Fund: DFAS, DFAT, DFLV, DFSV, DFUV
        r"^IJ[A-Z]$",       # iShares small-cap: IJH, IJK, IJR
        r"^VU[A-Z]$",       # Vanguard growth: VUG, VUO
        r"^VT[A-Z]{1,3}$",
        r"^VO[A-Z]{1,3}$",
        # International country funds (EW*)
        r"^EW[A-Z]{1,2}$",
        r"^VW[A-Z]{1,2}$",
        r"^IEV[A-Z]*$",
        r"^EE[A-Z]{1,3}$",
        # Vanguard funds families
        r"^V[OTW][A-Z]{1,3}$",
        # Commodities / metals
        r"^GLD[A-Z]*$",
        r"^SLV$",
        r"^IAU[A-Z]*$",
        r"^USO$",
        r"^UNG$",
        # Bonds
        r"^TLT$",
        r"^IEF$",
        r"^SHY$",
        r"^AGG$",
        r"^BND$",
        r"^LQD$",
        r"^HYG$",
        r"^TIP$",
        # Emerging markets / international
        r"^IEFA$",
        r"^IEMG$",
        r"^EEM$",
        r"^EFA$",
        r"^VEA$",
        r"^VWO$",
        r"^VXUS$",
    )
)

_ETF_KNOWN: frozenset[str] = frozenset(
    {
        "AGG",
        # Avantis family
        "AVDE",
        "AVEM",
        "AVUS",
        "BND",
        # Dimensional Fund family
        "DFAS",
        "DFAT",
        "DFLV",
        "DFSV",
        "DFUV",
        # WisdomTree family
        "DGRW",
        "DHS",
        # Broad market / style / size
        "DIA",
        # WisdomTree family
        "DLN",
        # Inverse/Leveraged ETFs
        "DOG",
        # WisdomTree family
        "DON",
        # Inverse/Leveraged ETFs
        "DXD",
        # iShares family
        "EEM",
        "EFA",
        "GLD",
        "HYG",
        # iShares Core series / family
        "IAU",
        "IBB",
        "IEF",
        "IEFA",
        "IEMG",
        "IJH",
        "IJK",
        "IJR",
        "ITOT",
        "IUSB",
        "IUSG",
        "IUSV",
        "IVV",
        "IWB",
        "IWM",
        "IWP",
        "IWR",
        "IWS",
        "IXUS",
        "LQD",
        "MDY",
        "MDYG",
        "MDYV",
        "MGK",
        # Inverse/Leveraged ETFs
        "PSQ",
        "QQQ",
        "QQQE",
        "QQQI",
        "QQQJ",
        "QQQM",
        "RSP",
        # Inverse/Leveraged ETFs
        "RWM",
        # Schwab family
        "SCHB",
        "SCHD",
        "SCHF",
        "SCHG",
        "SCHH",
        "SCHI",
        "SCHK",
        "SCHM",
        "SCHO",
        "SCHP",
        "SCHQ",
        "SCHR",
        "SCHV",
        "SCHX",
        "SCHZ",
        # Inverse/Leveraged ETFs
        "SDOW",
        "SDS",
        "SH",
        "SHY",
        "SLV",
        # SPDR family
        "SPBO",
        "SPHQ",
        "SPIB",
        "SPLV",
        "SPMD",
        "SPSM",
        "SPTM",
        "SPTS",
        # Inverse/Leveraged ETFs
        "SPXS",
        "SPXU",
        "SPY",
        "SPYG",
        "SPYV",
        # Inverse/Leveraged ETFs
        "SQQQ",
        "SSO",
        "TBT",
        "TIP",
        "TLT",
        "TNA",
        "TQQQ",
        "TWM",
        "TZA",
        "UNG",
        # Inverse/Leveraged ETFs
        "UPRO",
        "USO",
        # Inverse/Leveraged ETFs
        "UWM",
        # Vanguard family
        "VBK",
        "VBR",
        "VEA",
        "VIG",
        "VIGI",
        "VOO",
        "VT",
        "VTI",
        "VTV",
        "VUG",
        "VUO",
        "VWO",
        "VXUS",
        "VYM",
        "VYMI",
        "XBI",
        # 2024-2025 Income / Covered Call ETFs
        "JEPI",
        "JEPQ",
        "DIVO",
        "TSLY",
        "NVDY",
        "CONY",
        "MSTY",
        "IWMY",
        "XYLD",
        "QYLD",
        "RYLD",
        # 2024-2025 Thematic / Factor ETFs
        "COWZ",
        "SPLG",
        "FTEC",
        "FDIS",
        "FHLC",
        # Buffer / Defined Outcome ETFs
        "BUFR",
        "UJAN",
        "UJUL",
        # Crypto-adjacent ETFs
        "IBIT",
        "FBTC",
        "GBTC",
        "ETHE",
        "BITO",
        # Sector SPDR (XL*)
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
    }
)


def is_etf(symbol: str) -> bool:
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return False
    if normalized in _ETF_KNOWN:
        return True

    # Keep heuristics conservative: ETFs are typically 3-5 letter all-caps
    # tickers, but so are many equities; only apply known ETF family patterns.
    if not re.fullmatch(r"[A-Z]{3,5}", normalized):
        return False

    return any(pattern.match(normalized) for pattern in _ETF_PATTERNS)
