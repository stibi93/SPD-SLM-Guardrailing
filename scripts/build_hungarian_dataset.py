#!/usr/bin/env python3
"""Build diverse Hungarian SPD training and evaluation JSONL datasets."""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from long_form_examples import (  # noqa: E402
    LONG_MULTI,
    LONG_NEGATIVE,
    LONG_SINGLE,
    MAX_TEXT_LENGTH,
    assert_max_length,
)

from spd.categories import CATEGORIES

from spd.dedup import find_overlap, find_template_overlap, split_disjoint, text_key, validate_disjoint

# Banking context prefixes/suffixes for diversity
_PREFIXES = [
    "Üdv!",
    "Jó napot!",
    "Segítséget kérek:",
    "Sürgős kérdésem van:",
    "A mobilbankban írtam már, de",
    "Telefonon nem tudták megoldani,",
    "Kérem a segítségüket,",
    "Egy gyors kérdés:",
    "Mielőtt aláírom a szerződést,",
    "A fiókban azt mondták,",
]

_SUFFIXES = [
    "Köszönöm előre is a választ.",
    "Mikor kapok visszajelzést?",
    "Ez sürgős lenne.",
    "Tudnak segíteni online is?",
    "Hol tudom ezt intézni?",
    "Milyen dokumentumok kellenek hozzá?",
    "Van erre külön űrlap?",
    "A számlaszámom már megvan nálatok.",
]

_LONG_INTROS = [
    "Jó napot, már második napja próbálok online intézni egy egyszerűnek tűnő ügyet,",
    "Tisztelt Ügyfélszolgálat, telefonon nem kaptam egyértelmű választ, ezért írok ide is,",
    "Segítséget kérek, mert a mobilbankban minden lépésnél más hibát kapok,",
    "Üdvözlöm a bankot, a fiókban azt mondták, gyorsabb lesz neten, de én elakadtam,",
    "Sürgős ügyem lenne, mert holnap lejár egy határidő,",
    "Először próbáltam chaten, aztán telefonon, most ide fordulok,",
]

_LONG_MIDDLES = [
    "közben a partnerem is vár rám, mert közös döntés kellene,",
    "a netbank felülete számomra nem egyértelmű, főleg a dokumentumfeltöltésnél,",
    "nem vagyok biztos benne, hogy jó kedvezményezettet választottam,",
    "korábban más banknál pár perc alatt megoldottam ugyanezt,",
    "félek, hogy ha rossz adatot adok meg, késlekedni fog az utalás,",
    "a tranzakció összege nem túl nagy, de számomra fontos lenne időben rendezni,",
    "már frissítettem az appot és újraindítottam a telefont is,",
]

_LONG_CLOSINGS = [
    "Kérem, írják meg pontosan, milyen lépéseket tegyek most.",
    "Tudnak-e telefonon is végigvezetni, ha online nem megy?",
    "Mikor kapok visszaigazolást, ha ma benyújtom a kérelmet?",
    "Előre is köszönöm a gyors és érthető segítséget.",
    "Ha kell, szívesen bemásolom a hibaüzenet szövegét is.",
]

_NEGATIVE_TEXTS = [
    "Mennyi a számlaegyenlegem?",
    "Utalást szeretnék indítani a feleségemnek.",
    "Elvesztettem a bankkártyámat, mit tegyek?",
    "Mikor érkezik meg a hiteligazolás?",
    "Módosítani szeretném a napi limitet.",
    "Kérem a tranzakciós kimutatást az elmúlt hónapról.",
    "Hogyan tudok devizát váltani az appban?",
    "A lakáshitel kamatát szeretném fixálni.",
    "Nem értem a számlakivonaton szereplő tételt.",
    "Beállítanám az automatikus csoportos beszedést.",
    "Mikor jár le a betéti kamatperiódus?",
    "Szeretnék új számlát nyitni a gyerekemnek.",
    "A QR-kódos fizetés nem működik nálam.",
    "Kérem a bankgarancia másolatát.",
    "Hogyan tudok PIN-kódot módosítani?",
    "Van lehetőség kamatmentes időszakra a hitelkártyán?",
    "A céges számlán nem látom a tegnapi beérkezést.",
    "Szeretném lemondani a kártyabiztosítást.",
    "Mikor kapom meg az éves adóigazolást?",
    "Átutalás külföldre mennyi idő alatt érkezik meg?",
    "A mobilappban nem jelenik meg az új megbízás.",
    "Szeretnék időpontot foglalni a fiókba.",
    "Hogyan tudom lekérni a bankszámlaszám IBAN formátumát?",
    "Kérdésem van a folyószámlahitel kamatozásáról.",
    "Módosítani szeretném a számlacsomagomat olcsóbbra.",
    "Nem kaptam meg az SMS-t a tranzakció megerősítéséhez.",
    "Szeretném aktiválni az internetes vásárlási limitet.",
    "Mikor kerül jóváírásra a csoportos beszedés?",
    "Kérem a hitel törlesztési naptárát e-mailben.",
    "Hogyan tudok cégként POS terminált igényelni?",
    "A megtakarítási számlám kamata csökkent, miért?",
    "Szeretném lezárni a régi devizaszámlámat.",
    "Nem találom a számlanyitási szerződést.",
    "Kérem a banki referencialevelet a bérleti szerződéshez.",
    "Hogyan tudok készpénzt felvenni ATM-ből limit nélkül?",
    "A partnerem nem kapta meg az utalást, mit tegyek?",
    "Szeretném módosítani a hitel törlesztési napját.",
    "Van lehetőség ingyenes számlavezetésre fiataloknak?",
    "Kérem a banki igazolást a NAV felé.",
    "Hogyan tudok biztonságosan csaló e-mailt jelenteni?",
    "A kártyám lejár, mikor kapom az újat?",
    "Szeretnék betétet lekötöttek kamattal.",
    "Miért von le kétszer ugyanazt a díjat?",
    "Kérem a számlatörténet exportot Excelbe.",
    "Hogyan tudok két számlát összekapcsolni az appban?",
    "A céges netbank nem enged be, mit tegyek?",
    "Szeretném emelni az internetes vásárlási limitet.",
    "Mikor érkezik meg a SWIFT utalás?",
    "Kérem a folyószámlakivonatot PDF-ben.",
    "Hogyan tudok megbízást törölni a mobilbankban?",
    "A kamatozó betétem automatikusan megújul?",
    "Szeretnék tanácsadást a nyugdíjpénztári számlámhoz.",
    "Mi a teendő, ha gyanús tranzakciót látok?",
    "Kérem a banki igazolást a pályázathoz.",
    "Hogyan tudok új kedvezményezettet rögzíteni?",
    "A hitelkártya számlázási ciklus mikor zár?",
    "Szeretném csökkenteni a hitelkeretemet.",
    "Nem értem a devizaárfolyam számítását.",
    "Kérem a bankszámla zárásának menetét.",
    "Hogyan tudok cégként csoportos beszedést indítani?",
    "A megtakarítási cél számlám nem frissül.",
    "Szeretnék tanácsot a befektetési alap választásához.",
]

_CATEGORY_TEXTS: dict[str, list[str]] = {
    "ethnicity": [
        "Roma származású vagyok, a közösségi programhoz kérek hitelt.",
        "Mint magyarországi német, szeretnék adományozni a hazai egyesületnek.",
        "Szlovák őseim miatt érdekel a külhonimagyar támogatási hitel.",
        "Roma vagyok, és a családi vállalkozásom fejlesztéséhez kérek keretet.",
        "Olasz származású magyar állampolgár vagyok, devizaszámlát nyitnék.",
        "A roma származásom miatt külön adókedvezményt kérhetek?",
        "Székely származású vagyok, a szülőföldön nyitott számlámra utalnék.",
        "Német nemzetiségűként szeretnék pályázati támogatást utalni.",
        "Rutén származásom dokumentumával igazolhatom a kedvezményt?",
        "Mint roma családfő, közösségi fejlesztési hitelt szeretnék igényelni.",
        "Horvát származású vagyok, a horvát egyesület számlájára utalok.",
        "Szlovén nemzetiségű vagyok, külön számlát nyitnék az egyesületnek.",
        "A roma identitásom miatt kérek információt a mikrohitelről.",
        "Zsidó származású vagyok, a közösségi alap számlájára adományoznék.",
        "Szerb származású magyar vagyok, külföldi rokonoknak utalok.",
        "Roma származású vállalkozóként kérek kamatmentes hitelt.",
        "Görög származású vagyok, a görög katolikus egyházi számlára utalok.",
        "Lengyel őseim miatt érdekel a külhoni támogatási lehetőség.",
        "Mint roma fiatal, pályakezdő hitelt szeretnék igényelni.",
        "Ukrán származású vagyok, a rokonoknak hrivnyában utalnék.",
        "A szlovák nemzetiségű klub számlájára szeretnék havi támogatást küldeni.",
        "Roma származásom alapján jogosult vagyok speciális támogatásra?",
        "Bolgár származású vagyok, a bolgár közösség számlájára utalok.",
        "Mint magyarországi sváb, a helyi egyesületnek adományoznék.",
        "Roma vagyok, és a lakhatási támogatásomat a számlámra várnám.",
        "Örmény származású vagyok, a közösségi számlára utalnék.",
        "Szlovák nemzetiségűként igazolhatom a kedvezményes hitelt?",
        "Roma származásúként kérek tájékoztatást a vállalkozói hitelről.",
        "Török származású magyar vagyok, családi számlát nyitnék.",
        "A roma származásom miatt külön bankszámlát kell nyitnom?",
        "Szlovén nemzetiségű családból jövök, közös számlát nyitnánk.",
        "Roma származású vagyok, a roma önkormányzati számlára utalok.",
        "Német nemzetiségűként a helyi nemzetiségi önkormányzat számlájára utalnék.",
        "Szlovák származású vagyok, a rokonoknak euróban utalok.",
        "Roma vagyok, és a családi támogatásomat erre a számlára kérem.",
        "Horvát származásúként a horvát klub számlájára utalok havi tagdíjat.",
        "Rutén származású vagyok, külön számlát nyitnék a közösségnek.",
        "Roma származású vállalkozóként kérek fejlesztési támogatást.",
        "Görög származású vagyok, a görög egyesület számlájára adományoznék.",
        "Szlovén nemzetiségű vagyok, nemzetiségi kedvezményre vagyok jogosult?",
        "Roma származásom miatt kérek információt a mikrohitel kamatáról.",
        "Ukrán származású vagyok, a rokonoknak dollárban utalnék.",
        "Bolgár származású vagyok, a bolgár egyesület számlájára utalok.",
        "Roma vagyok, és a közösségi fejlesztési alap számlájára utalnék.",
        "Szlovák nemzetiségűként igazolhatom a számlanyitási kedvezményt?",
        "Roma származású fiatal vagyok, első lakáshitelhez kérek tanácsot.",
        "Örmény származású vagyok, a közösségi adomány számlájára utalok.",
        "Német származású vagyok, a sváb egyesület számlájára utalnék.",
        "Roma származásúként kérek tájékoztatást a vállalkozói számlacsomagról.",
        "Horvát származású vagyok, a horvát nemzetiségi számlára utalok.",
        "Szlovén származású vagyok, nemzetiségi támogatást utalnék.",
        "Roma vagyok, és a családi támogatásomat a megadott számlára kérem.",
        "Görög származású vagyok, a görög közösség számlájára adományoznék.",
        "Rutén származású vagyok, a rutén egyesület számlájára utalok.",
        "Roma származásom alapján külön hitelprogramra vagyok jogosult?",
        "Szlovák nemzetiségű vagyok, a nemzetiségi számlára utalnék.",
        "Roma származású vállalkozóként kérek kamatkedvezményes hitelt.",
        "Török származású vagyok, a családi számlára utalnék.",
        "Roma vagyok, és a közösségi program számlájára szeretnék utalni.",
        "Német nemzetiségű vagyok, a nemzetiségi önkormányzat számlájára utalok.",
    ],
    "political_opinion": [
        "Fidesz-szimpatizáns vagyok, a párt helyi szervezetének utalok.",
        "DK-tagként szeretném befizetni az éves tagdíjat.",
        "MSZP-s vagyok, a párt számlájára adományoznék.",
        "Jobbikos meggyőződésem miatt a helyi szervezetnek utalok.",
        "Momentum-tag vagyok, a párt számlájára utalnék.",
        "Liberális nézeteket vallok, civil szervezetnek adományoznék.",
        "Baloldali politikai nézeteim miatt kérek tájékoztatást az adományozásról.",
        "Jobboldali értékeket képviselve utalok a helyi alapítványnak.",
        "Zöld párt támogatója vagyok, a kampányszámlára utalok.",
        "Szocialista meggyőződésű vagyok, a párt számlájára adományoznék.",
        "Konzervatív politikai nézeteket vallok, a helyi szervezetnek utalok.",
        "LMP-s vagyok, a párt számlájára utalnék havi támogatást.",
        "Radikális baloldali nézeteket képviselve adományoznék.",
        "Fidesz-tag vagyok, a tagdíjat a megadott számlára utalnám.",
        "DK-s vagyok, a párt kampányszámlájára adományoznék.",
        "MSZP-tagként szeretném befizetni a tagdíjat.",
        "Jobbikos vagyok, a helyi szervezet számlájára utalok.",
        "Momentum-támogató vagyok, a párt számlájára utalnék.",
        "Liberális politikai nézeteket vallok, civil mozgalomnak adományoznék.",
        "Baloldali értékeket képviselve utalok a helyi alapítványnak.",
        "Jobboldali meggyőződésű vagyok, a helyi szervezetnek utalok.",
        "Zöld politikai nézeteket vallok, a kampányszámlára utalok.",
        "Szocialista vagyok, a párt számlájára adományoznék.",
        "Konzervatív értékeket képviselve utalok a helyi alapítványnak.",
        "LMP-támogató vagyok, a párt számlájára utalnék.",
        "Radikális jobboldali nézeteket vallok, a helyi szervezetnek utalok.",
        "Fidesz-szimpatizáns vagyok, a kampányszámlára adományoznék.",
        "DK-s meggyőződésem miatt a párt számlájára utalok.",
        "MSZP-s vagyok, a tagdíjat a megadott számlára utalnám.",
        "Jobbikos tag vagyok, a helyi szervezet számlájára utalok.",
        "Momentum-tagként szeretném befizetni a tagdíjat.",
        "Liberális értékeket képviselve adományoznék civil szervezetnek.",
        "Baloldali politikai nézeteket vallok, a helyi alapítványnak utalok.",
        "Jobboldali meggyőződésű vagyok, a helyi szervezet számlájára utalok.",
        "Zöld párt támogatója vagyok, a kampányszámlára utalnék.",
        "Szocialista nézeteket képviselve adományoznék a pártnak.",
        "Konzervatív politikai nézeteket vallok, a helyi szervezetnek utalok.",
        "LMP-s vagyok, a párt számlájára utalnék havi támogatást.",
        "Radikális baloldali meggyőződésem miatt adományoznék.",
        "Fidesz-tag vagyok, a tagdíjat a megadott számlára utalnám.",
        "DK-támogató vagyok, a párt kampányszámlájára adományoznék.",
        "MSZP-tag vagyok, a tagdíjat a megadott számlára utalnám.",
        "Jobbikos szimpatizáns vagyok, a helyi szervezetnek utalok.",
        "Momentum-szimpatizáns vagyok, a párt számlájára utalnék.",
        "Liberális meggyőződésű vagyok, civil mozgalomnak adományoznék.",
        "Baloldali értékeket képviselve utalok a helyi alapítványnak.",
        "Jobboldali politikai nézeteket vallok, a helyi szervezet számlájára utalok.",
        "Zöld értékeket képviselve adományoznék a kampányszámlára.",
        "Szocialista politikai nézeteket vallok, a párt számlájára utalok.",
        "Konzervatív meggyőződésem miatt a helyi szervezetnek utalok.",
        "LMP-támogató vagyok, a párt számlájára utalnék havi támogatást.",
        "Radikális jobboldali meggyőződésem miatt adományoznék.",
        "Fidesz-szimpatizáns vagyok, a helyi szervezet számlájára utalok.",
        "DK-tag vagyok, a tagdíjat a megadott számlára utalnám.",
        "MSZP-szimpatizáns vagyok, a kampányszámlára adományoznék.",
        "Jobbikos vagyok, a párt számlájára utalnék.",
        "Momentum-támogató vagyok, a kampányszámlára adományoznék.",
        "Liberális politikai nézeteket vallok, a civil szervezet számlájára utalok.",
        "Baloldali meggyőződésem miatt a helyi alapítványnak utalok.",
        "Jobboldali értékeket képviselve adományoznék a helyi szervezetnek.",
        "Zöld párt tagja vagyok, a tagdíjat a megadott számlára utalnám.",
    ],
    "religion_belief": [
        "Katolikus vagyok, az egyházam számlájára szeretnék adományozni.",
        "Református hitet vallok, a gyülekezet számlájára utalok.",
        "Evangélikus vagyok, a templom felújítására adományoznék.",
        "Görögkatolikus vagyok, az egyházi számlára utalnék.",
        "Unitárius hitet vallok, a gyülekezet számlájára adományoznék.",
        "Baptista vagyok, a misszió számlájára utalok.",
        "Metodista hitet vallok, az egyházi számlára utalnék.",
        "Zsidó vagyok, a zsinagóga számlájára adományoznék.",
        "Neológ zsidó hitet vallok, a közösség számlájára utalok.",
        "Ortodox zsidó vagyok, a hitközség számlájára utalnék.",
        "Muszlim vagyok, a mecset alapítvány számlájára adományoznék.",
        "Szunnita hitet vallok, a vallási közösség számlájára utalok.",
        "Buddhista vagyok, a templom számlájára adományoznék.",
        "Hindu hitet vallok, a közösség számlájára utalnék.",
        "Ateista világnézetet vallok, szkeptikus csoportnak adományoznék.",
        "Agnosztikus vagyok, humanista szervezetnek utalok.",
        "Szecularista nézeteket vallok, civil szervezetnek adományoznék.",
        "Katolikus családból jövök, az egyházi adományt utalnám.",
        "Református vagyok, a helyi gyülekezet számlájára utalok.",
        "Evangélikus hitet vallok, a templom számlájára adományoznék.",
        "Görögkatolikus vagyok, az egyházi számlára utalnék.",
        "Unitárius vagyok, a gyülekezet számlájára adományoznék.",
        "Baptista hitet vallok, a misszió számlájára utalok.",
        "Metodista vagyok, az egyházi számlára utalnék.",
        "Zsidó hitet vallok, a zsinagóga számlájára adományoznék.",
        "Neológ zsidó vagyok, a közösség számlájára utalok.",
        "Ortodox zsidó hitet vallok, a hitközség számlájára utalnék.",
        "Muszlim hitet vallok, a mecset alapítvány számlájára adományoznék.",
        "Szunnita vagyok, a vallási közösség számlájára utalok.",
        "Buddhista hitet vallok, a templom számlájára adományoznék.",
        "Hindu vagyok, a közösség számlájára utalnék.",
        "Ateista vagyok, szkeptikus csoportnak adományoznék.",
        "Agnosztikus világnézetet vallok, humanista szervezetnek utalok.",
        "Szecularista vagyok, civil szervezetnek adományoznék.",
        "Katolikus vagyok, a püspökség számlájára utalnék.",
        "Református hitet vallok, a diakóniai szolgálat számlájára adományoznék.",
        "Evangélikus vagyok, a gyülekezet számlájára utalok.",
        "Görögkatolikus hitet vallok, az egyházi számlára utalnék.",
        "Unitárius vagyok, a templom számlájára adományoznék.",
        "Baptista vagyok, a misszió számlájára utalok.",
        "Metodista hitet vallok, az egyházi számlára utalnék.",
        "Zsidó vagyok, a zsinagóga számlájára adományoznék.",
        "Neológ zsidó hitet vallok, a közösség számlájára utalok.",
        "Ortodox zsidó vagyok, a hitközség számlájára utalnék.",
        "Muszlim vagyok, a mecset alapítvány számlájára adományoznék.",
        "Szunnita hitet vallok, a vallási közösség számlájára utalok.",
        "Buddhista vagyok, a templom számlájára adományoznék.",
        "Hindu hitet vallok, a közösség számlájára utalnék.",
        "Ateista világnézetet vallok, szkeptikus csoportnak adományoznék.",
        "Agnosztikus vagyok, humanista szervezetnek utalok.",
        "Szecularista nézeteket vallok, civil szervezetnek adományoznék.",
        "Katolikus családból jövök, az egyházi adományt utalnám.",
        "Református vagyok, a helyi gyülekezet számlájára utalok.",
        "Evangélikus hitet vallok, a templom számlájára adományoznék.",
        "Görögkatolikus vagyok, az egyházi számlára utalnék.",
        "Unitárius hitet vallok, a gyülekezet számlájára adományoznék.",
        "Baptista vagyok, a misszió számlájára utalok.",
        "Metodista hitet vallok, az egyházi számlára utalnék.",
        "Zsidó hitet vallok, a zsinagóga számlájára adományoznék.",
        "Pünkösdista vagyok, a gyülekezet számlájára utalok.",
        "Kálvinista hitet vallok, a templom felújítására adományoznék.",
    ],
    "trade_union": [
        "A Vasas szakszervezet tagdíját szeretném utalni.",
        "Szakszervezeti tag vagyok, a havi tagdíjat befizetném.",
        "A szakszervezeti számlára utalnám az éves tagdíjat.",
        "Szakszervezeti tagságom miatt kérek tájékoztatást az utalásról.",
        "A MASZSZ tagdíját szeretném utalni a megadott számlára.",
        "Szakszervezeti tagként kérek bizonylatot az utalásról.",
        "A szakszervezeti tagság díját havi rendszerességgel utalnám.",
        "Szakszervezeti tag vagyok, a tagdíjat csoportos beszedéssel fizetném.",
        "A szakszervezeti számlára utalnám a sztrájk alap támogatását.",
        "Szakszervezeti tagságom igazolásához kérek utalási bizonylatot.",
        "A szakszervezeti tagdíjat a megadott számlára utalnám.",
        "Szakszervezeti tag vagyok, a havi díjat befizetném.",
        "A szakszervezeti számlára utalnám az éves tagdíjat.",
        "Szakszervezeti tagságom miatt kérek kedvezményt a számlacsomagra.",
        "A szakszervezeti tagdíjat csoportos beszedéssel fizetném.",
        "Szakszervezeti tag vagyok, a tagdíjat utalnám.",
        "A szakszervezeti számlára utalnám a sztrájk alap támogatását.",
        "Szakszervezeti tagságom igazolásához kérek bizonylatot.",
        "A szakszervezeti tagdíjat a megadott számlára utalnám.",
        "Szakszervezeti tag vagyok, a havi tagdíjat befizetném.",
        "A MASZSZ tagdíját szeretném utalni.",
        "Szakszervezeti tagságom miatt kérek tájékoztatást.",
        "A szakszervezeti számlára utalnám az éves tagdíjat.",
        "Szakszervezeti tagként kérek utalási bizonylatot.",
        "A szakszervezeti tagdíjat havi rendszerességgel utalnám.",
        "Szakszervezeti tag vagyok, csoportos beszedéssel fizetném.",
        "A szakszervezeti számlára utalnám a támogatást.",
        "Szakszervezeti tagságom igazolásához kérek bizonylatot az utalásról.",
        "A Vasas szakszervezet tagdíját utalnám.",
        "Szakszervezeti tag vagyok, a tagdíjat befizetném.",
        "A szakszervezeti számlára utalnám a havi tagdíjat.",
        "Szakszervezeti tagságom miatt kérek kedvezményt.",
        "A szakszervezeti tagdíjat csoportos beszedéssel fizetném.",
        "Szakszervezeti tag vagyok, az éves tagdíjat utalnám.",
        "A szakszervezeti számlára utalnám a sztrájk alap támogatását.",
        "Szakszervezeti tagságom igazolásához kérek utalási bizonylatot.",
        "A MASZSZ tagdíját a megadott számlára utalnám.",
        "Szakszervezeti tag vagyok, a havi díjat befizetném.",
        "A szakszervezeti számlára utalnám az éves tagdíjat.",
        "Szakszervezeti tagságom miatt kérek tájékoztatást az utalásról.",
        "A szakszervezeti tagdíjat utalnám a megadott számlára.",
        "Szakszervezeti tag vagyok, csoportos beszedéssel fizetném a tagdíjat.",
        "A szakszervezeti számlára utalnám a támogatást.",
        "Szakszervezeti tagságom igazolásához kérek bizonylatot.",
        "A Vasas szakszervezet tagdíját szeretném utalni.",
        "Szakszervezeti tag vagyok, a tagdíjat havi rendszerességgel utalnám.",
        "A szakszervezeti számlára utalnám az éves tagdíjat.",
        "Szakszervezeti tagságom miatt kérek kedvezményt a számlacsomagra.",
        "A szakszervezeti tagdíjat csoportos beszedéssel fizetném.",
        "Szakszervezeti tag vagyok, a tagdíjat befizetném.",
        "A szakszervezeti számlára utalnám a sztrájk alap támogatását.",
        "Szakszervezeti tagságom igazolásához kérek utalási bizonylatot.",
        "A MASZSZ tagdíját utalnám.",
        "Szakszervezeti tag vagyok, a havi tagdíjat befizetném.",
        "A szakszervezeti számlára utalnám a tagdíjat.",
        "Szakszervezeti tagságom miatt kérek tájékoztatást.",
        "A szakszervezeti tagdíjat a megadott számlára utalnám.",
        "Szakszervezeti tag vagyok, az éves tagdíjat utalnám.",
        "A szakszervezeti számlára utalnám a havi tagdíjat.",
        "Szakszervezeti tagságom igazolásához kérek bizonylatot az utalásról.",
    ],
    "genetic": [
        "A BRCA1 mutációm miatt kérek biztosítási fedezetet.",
        "Genetikai tesztem eredménye alapján igazolom az örökletes kockázatot.",
        "Örökletes cukorbetegség van a családomban, genetikai szűrésen estem át.",
        "A DNS-tesztem mutációt mutatott, ehhez kapcsolódó hitelt kérek.",
        "Genetikai vizsgálatom alapján magas a szívinfarktus kockázatom.",
        "Örökletes rák szindrómám van, genetikai tanácsadást kaptam.",
        "A genetikai teszt pozitív lett a Huntington-kórra.",
        "Családi genetikai szűrésen vettem részt, eredményt csatolok.",
        "Örökletes vérzékenységem genetikai okokra vezethető vissza.",
        "Genetikai panel eredménye alapján kérek biztosítási felmérést.",
        "A BRCA2 génmutációm miatt speciális hitelkonstrukciót keresek.",
        "Genetikai teszt igazolja az örökletes betegség kockázatom.",
        "Örökletes magas koleszterin szintem genetikai eredetű.",
        "A DNS-vizsgálat mutációt talált a tumorra hajlamosító génben.",
        "Genetikai szűrésen pozitív lett az örökletes neuropátiára.",
        "Örökletes vesebetegség van a családban, genetikai tesztet végeztem.",
        "Genetikai tanácsadás után kaptam írásos eredményt a banknak.",
        "A genetikai vizsgálat alapján magas a melanoma kockázatom.",
        "Örökletes trombofíliám genetikai teszttel igazolt.",
        "BRCA mutáció miatt kérek kedvezményes biztosítási csomagot.",
        "Genetikai tesztem alapján örökletes szívbetegség kockázatom van.",
        "A családi genetikai szűrés pozitív eredményt hozott.",
        "Örökletes autoimmun betegségem genetikai vizsgálattal igazolt.",
        "Genetikai panel mutációt talált az örökletes rák szindrómában.",
        "DNS-teszt eredménye alapján kérek hiteligazolást.",
        "Genetikai szűrésen pozitív lett az örökletes cukorbetegségre.",
        "Örökletes vérnyomás-problémám genetikai okokra vezethető vissza.",
        "A genetikai vizsgálat mutációt mutatott a CFTR génben.",
        "Genetikai tanácsadás után kaptam írásos jelentést.",
        "Örökletes epilepsia van a családban, genetikai tesztet végeztem.",
        "BRCA1 pozitív genetikai tesztem van, ehhez kapcsolódó kérelmet nyújtok.",
        "Genetikai szűrés alapján magas az örökletes Alzheimer-kockázatom.",
        "Örökletes vérszegénységem genetikai vizsgálattal igazolt.",
        "A DNS-teszt mutációt talált az örökletes vesebetegségben.",
        "Genetikai panel eredménye alapján kérek biztosítási tanácsot.",
        "Örökletes rák hajlamom genetikai teszttel igazolt.",
        "Genetikai vizsgálatom pozitív lett a Lynch-szindrómára.",
        "A családi genetikai szűrés mutációt talált.",
        "Örökletes szívbillentyű-betegségem genetikai okokra vezethető vissza.",
        "Genetikai teszt eredménye alapján kérek kedvezményes hitelt.",
        "BRCA2 mutációm miatt speciális biztosítási fedezetet keresek.",
        "Genetikai szűrésen pozitív lett az örökletes vérzékenységre.",
        "Örökletes cukorbetegség kockázatom genetikai teszttel igazolt.",
        "A genetikai vizsgálat mutációt mutatott az örökletes rákban.",
        "Genetikai tanácsadás után kaptam írásos eredményt.",
        "Örökletes magas koleszterin szintem genetikai eredetű.",
        "DNS-tesztem alapján magas a szívinfarktus kockázatom.",
        "Genetikai panel pozitív eredményt hozott az örökletes betegségre.",
        "Örökletes neuropátiám genetikai vizsgálattal igazolt.",
        "A genetikai teszt mutációt talált a tumorra hajlamosító génben.",
        "Genetikai szűrés alapján kérek biztosítási felmérést.",
        "Örökletes vesebetegség kockázatom genetikai teszttel igazolt.",
        "BRCA mutáció miatt kérek speciális hitelkonstrukciót.",
        "Genetikai vizsgálatom pozitív lett az örökletes rák szindrómára.",
        "A családi genetikai szűrés alapján magas a melanoma kockázatom.",
        "Örökletes trombofíliám genetikai teszttel igazolt.",
        "Genetikai panel eredménye alapján kérek hiteligazolást.",
        "Örökletes autoimmun betegség kockázatom genetikai vizsgálattal igazolt.",
        "A DNS-teszt pozitív lett az örökletes cukorbetegségre.",
        "Genetikai tanácsadás után kaptam írásos jelentést a banknak.",
    ],
    "health": [
        "Cukorbeteg vagyok, hitelt szeretnék felvenni.",
        "Rákos betegségem miatt kérek halasztást a törlesztésre.",
        "Magas vérnyomásom van, gyógyszert szedek.",
        "Asztmás vagyok, orvosi igazolást csatolok.",
        "Depresszióval kezeltetem magam, pszichiátriai igazolást küldök.",
        "Szívinfarktusom volt, kardiológiai leletet csatolok.",
        "Vesebetegségem miatt dialízist kapok.",
        "Epilepsziás vagyok, orvosi igazolást csatolok.",
        "Krónikus hátfájdalom miatt rokkantsági ellátást kapok.",
        "Allergiás vagyok penicillinre, orvosi figyelmeztetést csatolok.",
        "Pszichés betegségem miatt kérek halasztást a hitel törlesztésére.",
        "Cukorbetegségem miatt speciális biztosítást keresek.",
        "Rák miatt kezelés alatt állok, orvosi igazolást csatolok.",
        "Magas koleszterin szintem van, gyógyszert szedek.",
        "Asztmás rohamaim vannak, orvosi igazolást küldök.",
        "Depresszió miatt antidepresszánsokat szedek.",
        "Szívbetegségem miatt kardiológiai kezelés alatt állok.",
        "Veseelégtelenségem miatt dialízist kapok.",
        "Epilepsziás rohamaim vannak, orvosi igazolást csatolok.",
        "Krónikus fájdalom miatt rokkantsági ellátást kapok.",
        "Súlyos allergiám van, orvosi figyelmeztetést csatolok.",
        "Pszichés zavarom miatt kérek halasztást.",
        "Cukorbeteg vagyok, inzulint használok.",
        "Rákos megbetegedésem miatt kemoterápiát kapok.",
        "Magas vérnyomásom miatt gyógyszert szedek.",
        "Asztmás vagyok, inhalátort használok.",
        "Depresszióval kezeltetem magam, pszichiátriai kezelés alatt állok.",
        "Szívinfarktusom után kardiológiai kontrollon vagyok.",
        "Vesebetegségem miatt orvosi kezelés alatt állok.",
        "Epilepsziás vagyok, gyógyszert szedek.",
        "Krónikus betegségem miatt rokkantsági ellátást kapok.",
        "Allergiás reakcióim vannak, orvosi igazolást csatolok.",
        "Pszichés betegségem miatt kérek halasztást a törlesztésre.",
        "Cukorbetegségem miatt orvosi igazolást csatolok.",
        "Rák miatt műtéten estem át, orvosi leletet küldök.",
        "Magas koleszterin szintem miatt gyógyszert szedek.",
        "Asztmás rohamaim miatt orvosi igazolást csatolok.",
        "Depresszió miatt pszichiátriai kezelés alatt állok.",
        "Szívbetegségem miatt pacemakerem van.",
        "Veseelégtelenségem miatt orvosi kezelés alatt állok.",
        "Epilepsziás rohamaim miatt orvosi igazolást küldök.",
        "Krónikus fájdalom miatt fájdalomcsillapítót szedek.",
        "Súlyos allergiám miatt orvosi figyelmeztetést csatolok.",
        "Pszichés zavarom miatt kérek halasztást a hitel törlesztésére.",
        "Cukorbeteg vagyok, és hitelt szeretnék felvenni.",
        "Rákos betegségem miatt kezelés alatt állok.",
        "Magas vérnyomásom van, orvosi igazolást csatolok.",
        "Asztmás vagyok, orvosi kezelés alatt állok.",
        "Depresszióval kezeltetem magam, orvosi igazolást küldök.",
        "Szívinfarktusom miatt kardiológiai kezelés alatt állok.",
        "Vesebetegségem miatt dialízist kapok, orvosi igazolást csatolok.",
        "Epilepsziás vagyok, orvosi kezelés alatt állok.",
        "Krónikus betegségem miatt rokkantsági ellátást kapok.",
        "Allergiás vagyok, orvosi figyelmeztetést csatolok.",
        "Pszichés betegségem miatt kérek halasztást a törlesztésre.",
        "Cukorbetegségem miatt speciális hitelkonstrukciót keresek.",
        "Rák miatt orvosi kezelés alatt állok, igazolást csatolok.",
        "Magas vérnyomásom miatt gyógyszert szedek, orvosi igazolást küldök.",
        "Asztmás rohamaim miatt orvosi kezelés alatt állok.",
        "Depresszió miatt orvosi igazolást csatolok.",
        "Szívbetegségem miatt orvosi kezelés alatt állok.",
    ],
    "sex_life_orientation": [
        "Meleg vagyok, és a párommal közös számlát nyitnánk.",
        "Leszbikus vagyok, hitelt szeretnénk felvenni a párommal.",
        "Biszexuális vagyok, közös számlát nyitnánk a partneremmel.",
        "Transznemű vagyok, névváltoztatás után módosítanám a számlát.",
        "Az LMBTQ közösség számlájára szeretnék adományozni.",
        "Meleg párommal közös hitelt szeretnénk igényelni.",
        "Leszbikus vagyok, és a párommal közös számlát nyitnánk.",
        "Biszexuális vagyok, hitelt szeretnék felvenni a partneremmel.",
        "Transznemű vagyok, dokumentumok frissítése után módosítanám a számlát.",
        "Az LMBTQ alapítvány számlájára adományoznék.",
        "Meleg vagyok, közös számlát nyitnánk a párommal.",
        "Leszbikus párommal hitelt szeretnénk igényelni.",
        "Biszexuális vagyok, közös számlát nyitnánk.",
        "Transznemű vagyok, névváltoztatás miatt kérek számlamódosítást.",
        "Az LMBTQ közösség támogatására utalnék.",
        "Meleg párommal közös számlát nyitnánk.",
        "Leszbikus vagyok, hitelt szeretnék felvenni.",
        "Biszexuális vagyok, a partneremmel közös hitelt kérnénk.",
        "Transznemű vagyok, dokumentumok frissítése után módosítanám az adataimat.",
        "Az LMBTQ szervezet számlájára adományoznék.",
        "Meleg vagyok, és hitelt szeretnék felvenni a párommal.",
        "Leszbikus párommal közös számlát nyitnánk.",
        "Biszexuális vagyok, közös hitelt szeretnénk igényelni.",
        "Transznemű vagyok, névváltoztatás után frissíteném a számlát.",
        "Az LMBTQ alap számlájára utalnék.",
        "Meleg párommal hitelt szeretnénk igényelni.",
        "Leszbikus vagyok, közös számlát nyitnánk a párommal.",
        "Biszexuális vagyok, hitelt szeretnék felvenni.",
        "Transznemű vagyok, dokumentumok miatt kérek számlamódosítást.",
        "Az LMBTQ közösség számlájára adományoznék.",
        "Meleg vagyok, közös hitelt szeretnénk igényelni a párommal.",
        "Leszbikus párommal hitelt szeretnénk felvenni.",
        "Biszexuális vagyok, közös számlát nyitnánk a partneremmel.",
        "Transznemű vagyok, névváltoztatás miatt módosítanám a számlát.",
        "Az LMBTQ támogatására utalnék.",
        "Meleg párommal közös számlát nyitnánk.",
        "Leszbikus vagyok, és hitelt szeretnék felvenni a párommal.",
        "Biszexuális vagyok, közös hitelt kérnénk.",
        "Transznemű vagyok, dokumentumok frissítése után módosítanám a számlát.",
        "Az LMBTQ alapítvány számlájára utalnék.",
        "Meleg vagyok, hitelt szeretnék felvenni.",
        "Leszbikus párommal közös számlát nyitnánk.",
        "Biszexuális vagyok, a partneremmel hitelt szeretnénk igényelni.",
        "Transznemű vagyok, névváltoztatás után kérek adatmódosítást.",
        "Az LMBTQ szervezet támogatására adományoznék.",
        "Meleg párommal hitelt szeretnénk felvenni.",
        "Leszbikus vagyok, közös hitelt szeretnénk igényelni.",
        "Biszexuális vagyok, közös számlát nyitnánk.",
        "Transznemű vagyok, dokumentumok miatt frissíteném a számlát.",
        "Az LMBTQ közösség alap számlájára utalnék.",
        "Meleg vagyok, és közös számlát nyitnánk a párommal.",
        "Leszbikus párommal hitelt szeretnénk igényelni.",
        "Biszexuális vagyok, hitelt szeretnék felvenni a partneremmel.",
        "Transznemű vagyok, névváltoztatás miatt módosítanám az adataimat.",
        "Az LMBTQ alap számlájára adományoznék.",
        "Meleg párommal közös hitelt szeretnénk igényelni.",
        "Leszbikus vagyok, hitelt szeretnék felvenni a párommal.",
        "Biszexuális vagyok, közös hitelt kérnénk a partneremmel.",
        "Transznemű vagyok, dokumentumok frissítése után módosítanám a számlát.",
        "Az LMBTQ támogatására utalnék a megadott számlára.",
    ],
}


# (text, list of positive categories) — realistic multi-SPD banking chat messages
_MULTI_LABEL_EXAMPLES: list[tuple[str, list[str]]] = [
    (
        "Van egy lezbikus párom és szeretnénk közös számlát nyitni, de látássérülésem "
        "miatt nem tudok normálisan elnavigálni a honlapon. Segítség!",
        ["sex_life_orientation", "health"],
    ),
    (
        "Nem sikerült tegnap letiltanom a kártyát. Van egy lezbikus párom és szeretnénk "
        "majd közös számlát nyitni, viszont látássérülésem miatt nem tudok normálisan "
        "elnavigálni a honlapra. Segítség!",
        ["sex_life_orientation", "health"],
    ),
    (
        "Meleg vagyok, cukorbetegségem van, és közös hitelt szeretnénk a párommal — "
        "orvosi igazolást is csatolok.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Biszexuális párommal közös számlát nyitnánk; epilepsziás vagyok, "
        "orvosi dokumentumot küldök.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Transznemű vagyok, depresszióval kezeltetem magam, névváltoztatás után "
        "frissíteném a bankszámlámat.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Leszbikus vagyok, kerekesszékes vagyok, és telefonon szeretném intézni "
        "a közös számlanyitást.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Meleg párommal hitelt vennénk fel; szívbetegségem miatt halasztást kérek "
        "a törlesztésre.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Van egy meleg párom, rákos vagyok, és közös számlát szeretnénk nyitni "
        "a kezelésem idejére.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Biszexuális vagyok, halláskárosodásom van, ezért videós azonosítás helyett "
        "személyesen szeretnék számlát nyitni.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Leszbikus párommal közös számlát nyitnánk; asztmás vagyok, "
        "inhalátorom miatt utazási biztosítást is keresek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Katolikus vagyok és cukorbeteg; az egyház számlájára adományoznék "
        "orvosi kezelésem költségei miatt részletfizetést kérek.",
        ["religion_belief", "health"],
    ),
    (
        "Református hitet vallok, szívinfarktusom volt, templomi adományt "
        "és hitelhalasztást is szeretnék intézni.",
        ["religion_belief", "health"],
    ),
    (
        "Zsidó vagyok, krónikus vesebetegségem van, a hitközség számlájára "
        "és dialízisre is utalást kérek.",
        ["religion_belief", "health"],
    ),
    (
        "Muszlim vagyok, allergiás vagyok penicillinre, a mecset alapítványnak "
        "utalnék és orvosi igazolást csatolok.",
        ["religion_belief", "health"],
    ),
    (
        "Ateista világnézetet vallok, depresszió miatt kezelés alatt állok, "
        "humanista szervezetnek adományoznék.",
        ["religion_belief", "health"],
    ),
    (
        "Roma származású vagyok, cukorbeteg vagyok, közösségi hitelt "
        "és egészségügyi támogatást kérek.",
        ["ethnicity", "health"],
    ),
    (
        "Német nemzetiségű vagyok, rákos megbetegedésem miatt a nemzetiségi "
        "számlára utalnék és halasztást kérek.",
        ["ethnicity", "health"],
    ),
    (
        "Szlovák származású vagyok, epilepsziás vagyok, rokonoknak utalnék "
        "és orvosi igazolást küldök.",
        ["ethnicity", "health"],
    ),
    (
        "Roma vagyok, Fidesz-szimpatizáns vagyok, közösségi és párt számlára "
        "is utalnék.",
        ["ethnicity", "political_opinion"],
    ),
    (
        "DK-tag vagyok, roma származású vagyok, tagdíjat és közösségi "
        "támogatást utalnék.",
        ["ethnicity", "political_opinion"],
    ),
    (
        "MSZP-s vagyok, szlovák nemzetiségű vagyok, párt és nemzetiségi "
        "számlára utalok.",
        ["ethnicity", "political_opinion"],
    ),
    (
        "Szakszervezeti tag vagyok, baloldali nézeteket vallok, tagdíjat "
        "és adományt utalnék.",
        ["trade_union", "political_opinion"],
    ),
    (
        "Vasas szakszervezeti tag vagyok, zöld párt támogatója vagyok, "
        "mindkét számlára utalnék.",
        ["trade_union", "political_opinion"],
    ),
    (
        "Szakszervezeti tagdíjat fizetek, konzervatív értékeket vallok, "
        "két külön utalást szeretnék indítani.",
        ["trade_union", "political_opinion"],
    ),
    (
        "BRCA mutációm van, cukorbeteg vagyok, genetikai és egészségügyi "
        "dokumentumot csatolok a hitelhez.",
        ["genetic", "health"],
    ),
    (
        "Örökletes szívbetegségem genetikai teszttel igazolt, magas vérnyomásom "
        "van, biztosítást kérek.",
        ["genetic", "health"],
    ),
    (
        "Genetikai vizsgálatom pozitív lett, rákos vagyok, orvosi és genetikai "
        "leletet küldök.",
        ["genetic", "health"],
    ),
    (
        "Örökletes cukorbetegség van a családban, magam is cukorbeteg vagyok, "
        "DNS-teszt eredményt csatolok.",
        ["genetic", "health"],
    ),
    (
        "Látássérülésem miatt nem találom a megfelelő menüpontot, kerekesszékes "
        "vagyok, és a fiókban szeretném beállítani az akadálymentes ügyintézést.",
        ["health"],
    ),
    (
        "Vak vagyok, segítséget kérek a netbank használatához, mert a felület "
        "nem akadálymentes és elakadtam a számlanyitásnál.",
        ["health"],
    ),
    (
        "Kerekesszékes vagyok, a fiókban szeretném intézni a hitelhalasztást "
        "és az akadálymentes bejutást is.",
        ["health"],
    ),
    (
        "Tremorom miatt nehezen használom a touchpadet — Parkinson-kóros vagyok, "
        "telefonos ügyintézést kérek.",
        ["health"],
    ),
    (
        "Katolikus vagyok, roma származású vagyok, egyházi és közösségi "
        "számlára utalnék.",
        ["religion_belief", "ethnicity"],
    ),
    (
        "Református vagyok, német nemzetiségű vagyok, templomnak és "
        "nemzetiségi szervezetnek adományoznék.",
        ["religion_belief", "ethnicity"],
    ),
    (
        "Zsidó vagyok, szlovák származású vagyok, hitközségnek és rokonoknak utalok.",
        ["religion_belief", "ethnicity"],
    ),
    (
        "Muszlim vagyok, török származású vagyok, mecset alapítványnak utalnék.",
        ["religion_belief", "ethnicity"],
    ),
    (
        "Meleg vagyok, katolikus vagyok, közös számlát nyitnánk a párommal "
        "és az egyháznak adományoznék.",
        ["sex_life_orientation", "religion_belief"],
    ),
    (
        "Leszbikus vagyok, református hitet vallok, gyülekezetnek adományoznék "
        "és közös hitelt kérnénk.",
        ["sex_life_orientation", "religion_belief"],
    ),
    (
        "Biszexuális vagyok, zsidó vagyok, LMBTQ és hitközségi számlára utalnék.",
        ["sex_life_orientation", "religion_belief"],
    ),
    (
        "Transznemű vagyok, buddhista vagyok, templom számlájára adományoznék.",
        ["sex_life_orientation", "religion_belief"],
    ),
    (
        "Meleg vagyok, roma származású vagyok, közösségi számlára és "
        "közös páros számlára is érdeklődöm.",
        ["sex_life_orientation", "ethnicity"],
    ),
    (
        "Leszbikus párommal számlát nyitnánk, szlovák nemzetiségű vagyok, "
        "két külön utalást indítanék.",
        ["sex_life_orientation", "ethnicity"],
    ),
    (
        "Biszexuális vagyok, DK-tag vagyok, párt számlájára és közös "
        "számlára utalnék.",
        ["sex_life_orientation", "political_opinion"],
    ),
    (
        "Meleg párommal hitelt vennénk, Momentum-tag vagyok, kampányszámlára "
        "is adományoznék.",
        ["sex_life_orientation", "political_opinion"],
    ),
    (
        "Leszbikus vagyok, szakszervezeti tag vagyok, tagdíjat és "
        "közös számlát szeretnénk intézni.",
        ["sex_life_orientation", "trade_union"],
    ),
    (
        "Meleg vagyok, szakszervezeti tagdíjat fizetek, közös számlát "
        "nyitnánk a párommal.",
        ["sex_life_orientation", "trade_union"],
    ),
    (
        "BRCA1 mutációm van, leszbikus vagyok, genetikai és közös "
        "számlanyitási kérelmet nyújtok.",
        ["genetic", "sex_life_orientation"],
    ),
    (
        "Örökletes rákhajlamom van, meleg párommal biztosítást keresünk.",
        ["genetic", "sex_life_orientation"],
    ),
    (
        "Genetikai tesztem pozitív, transznemű vagyok, dokumentumok "
        "frissítése után hitelt kérek.",
        ["genetic", "sex_life_orientation"],
    ),
    (
        "Meleg vagyok, párommal közös számlát nyitnánk, látássérülésem miatt "
        "telefonos segítséget kérek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Leszbikus párommal közös számlát nyitnánk, hallássérült vagyok, "
        "jelnyelvi tolmácsos ügyintézést kérek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Szakszervezeti tag vagyok, cukorbeteg vagyok, tagdíjat utalnék "
        "és orvosi igazolást csatolok.",
        ["trade_union", "health"],
    ),
    (
        "Vasas szakszervezeti tag vagyok, depresszióval kezeltetem magam, "
        "halasztást kérek a hitelre.",
        ["trade_union", "health"],
    ),
    (
        "Szakszervezeti tagdíjat fizetek, asztmás vagyok, orvosi papírt küldök.",
        ["trade_union", "health"],
    ),
    (
        "Fidesz-tag vagyok, cukorbeteg vagyok, tagdíjat és orvosi igazolást küldök.",
        ["political_opinion", "health"],
    ),
    (
        "DK-s vagyok, szívbeteg vagyok, adományt és halasztást kérek.",
        ["political_opinion", "health"],
    ),
    (
        "Jobbikos vagyok, vesebeteg vagyok, párt számlájára utalnék.",
        ["political_opinion", "health"],
    ),
    (
        "Roma vagyok, szakszervezeti tag vagyok, közösségi és szakszervezeti "
        "számlára utalnék.",
        ["ethnicity", "trade_union"],
    ),
    (
        "Német nemzetiségű vagyok, szakszervezeti tag vagyok, két utalást indítanék.",
        ["ethnicity", "trade_union"],
    ),
    (
        "Katolikus vagyok, szakszervezeti tag vagyok, egyháznak és szakszervezetnek utalok.",
        ["religion_belief", "trade_union"],
    ),
    (
        "Református vagyok, Vasas tag vagyok, templomnak és szakszervezetnek adományoznék.",
        ["religion_belief", "trade_union"],
    ),
    (
        "BRCA mutációm és látássérülésem is van, genetikai és akadálymentesítési "
        "támogatást kérek.",
        ["genetic", "health"],
    ),
    (
        "Meleg vagyok, roma származású vagyok, cukorbeteg vagyok — közös számlát "
        "és közösségi támogatást kérek.",
        ["sex_life_orientation", "ethnicity", "health"],
    ),
    (
        "Leszbikus vagyok, katolikus vagyok, depresszióval kezeltetem magam, "
        "három külön kérelmet szeretnék benyújtani.",
        ["sex_life_orientation", "religion_belief", "health"],
    ),
    (
        "Transznemű vagyok, zöld párt támogatója vagyok, epilepsziás vagyok, "
        "dokumentumfrissítést és halasztást kérek.",
        ["sex_life_orientation", "political_opinion", "health"],
    ),
    (
        "Szlovák származású vagyok, örökletes cukorbetegségem van, genetikai "
        "leletet és rokoni utalást intéznék.",
        ["ethnicity", "genetic", "health"],
    ),
    (
        "Vak vagyok, meleg párommal közös számlát nyitnánk telefonos "
        "segítséggel, mert a felület nem akadálymentes.",
        ["health", "sex_life_orientation"],
    ),
    (
        "Nem sikerült tegnap letiltanom a kárVan egy lezbikus párom és szeretnénk majd "
        "közös számlát nyitni viszont van a látássérülésem miatt nem tudok nomálrisan "
        "elnavigálni a honlapra oda, ahol ezt meg tudom tenni. Segítség!",
        ["sex_life_orientation", "health"],
    ),
    (
        "Van egy lezbikus párom és szeretnénk majd közös számlát nyitni, viszont "
        "a látássérülésem miatt nem tudok normálisan elnavigálni a honlapra, "
        "ahol ezt meg tudom tenni. Segítség!",
        ["sex_life_orientation", "health"],
    ),
    (
        "Hallássérült meleg vagyok, párommal közös számlát szeretnénk, "
        "jelnyelvi tolmácsos ügyintézést kérek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Kerekesszékes leszbikus vagyok, párommal hitelt vennénk, "
        "akadálymentes fiókirodát keresek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Autista vagyok, biszexuális párommal közös számlát nyitnánk, "
        "egyszerűsített felületet kérek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Down-szindrómás rokom nevében utalok, meleg vagyok, közös számlát "
        "is nyitnánk a párommal.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Pszichés zavarom van, transznemű vagyok, névváltoztatás után "
        "frissíteném az adataimat és halasztást kérek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Krónikus fájdalom miatt rokkantsági ellátást kapok, leszbikus párommal "
        "közös számlát nyitnánk.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Súlyos allergiám van, meleg párommal utazási biztosítást és "
        "közös számlát szeretnénk.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Veseelégtelenségem miatt dialízist kapok, biszexuális vagyok, "
        "közös számlát nyitnánk a partneremmel.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Szívbeteg meleg vagyok, párommal közös hitelt kérnénk, "
        "kardiológiai igazolást csatolok.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Rákos meleg vagyok, kezelés alatt állok, párommal közös számlát "
        "szeretnénk a kórházi kiadások miatt.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Epilepsziás leszbikus vagyok, párommal számlát nyitnánk, "
        "orvosi igazolást mellékelek.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Asztmás biszexuális vagyok, partneremmel közös számlát és "
        "biztosítást keresünk.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Depresszió miatt kezelés alatt állok, meleg párommal közös számlát "
        "nyitnánk.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Magas vérnyomásom van, transznemű vagyok, dokumentumfrissítés "
        "és közös számla is kellene.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Cukorbeteg leszbikus vagyok, párommal közös számlát nyitnánk, "
        "inzulin-költségek miatt.",
        ["sex_life_orientation", "health"],
    ),
    (
        "Krónikus vesebetegségem és leszbikus párommal közös számlanyitás "
        "együtt kellene, de mindkét ügy külön űrlapon akadt el.",
        ["health", "sex_life_orientation"],
    ),
    (
        "Roma származású zöld párt támogató vagyok, két külön utalást "
        "szeretnék ma indítani.",
        ["ethnicity", "political_opinion"],
    ),
    (
        "Szakszervezeti tag és evangélikus vagyok, templomnak és "
        "szakszervezetnek utalnék.",
        ["trade_union", "religion_belief"],
    ),
    (
        "Örökletes magas koleszterin és asztmás rohamaim vannak, "
        "mindkettőről van orvosi papírom.",
        ["genetic", "health"],
    ),
    (
        "Meleg párommal hitelt vennénk, DK-tag vagyok, pártadományt "
        "is utalnék ma.",
        ["sex_life_orientation", "political_opinion"],
    ),
    (
        "Muszlim vagyok, szakszervezeti tag is, mecsetnek és "
        "szakszervezetnek utalok.",
        ["religion_belief", "trade_union"],
    ),
    (
        "Szlovák származású leszbikus vagyok, rokonoknak és "
        "párommal közös számlát is nyitnánk.",
        ["ethnicity", "sex_life_orientation"],
    ),
    (
        "BRCA mutációm és depresszióm is dokumentált, genetikai "
        "és pszichiátriai leletet csatolok.",
        ["genetic", "health"],
    ),
    (
        "Katolikus MSZP-s vagyok, pártnak és templomnak is "
        "szeretnék utalni.",
        ["religion_belief", "political_opinion"],
    ),
    (
        "Transznemű roma vagyok, névváltoztatás után frissíteném "
        "a számlát és közösségi támogatást utalnék.",
        ["sex_life_orientation", "ethnicity"],
    ),
    (
        "Hallássérült Fidesz-szimpatizáns vagyok, kampányszámlára "
        "utalnék jelnyelvi segítséggel.",
        ["health", "political_opinion"],
    ),
]


def _make_labels(positive: str | list[str] | None) -> dict[str, int]:
    labels = {cat: 0 for cat in CATEGORIES}
    if not positive:
        return labels
    cats = [positive] if isinstance(positive, str) else positive
    for cat in cats:
        if cat not in labels:
            raise ValueError(f"Unknown category label: {cat}")
        labels[cat] = 1
    return labels


def _clamp_text(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_TEXT_LENGTH:
        return text
    trimmed = text[:MAX_TEXT_LENGTH]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed.rstrip(" ,;:") + "."


def _pad_to_long(core: str, rng: random.Random, min_len: int = 320) -> str:
    parts = [rng.choice(_LONG_INTROS), core]
    while len(" ".join(parts)) < min_len:
        parts.insert(-1, rng.choice(_LONG_MIDDLES))
        if len(" ".join(parts)) >= MAX_TEXT_LENGTH - 90:
            break
    parts.append(rng.choice(_LONG_CLOSINGS))
    return _clamp_text(" ".join(parts))


def _diversify(text: str, rng: random.Random, add_context: bool) -> str:
    if not add_context:
        return text
    prefix = rng.choice(_PREFIXES)
    suffix = rng.choice(_SUFFIXES)
    style = rng.randint(0, 2)
    if style == 0:
        return f"{prefix} {text}"
    if style == 1:
        return _clamp_text(f"{text} {suffix}")
    return _clamp_text(f"{prefix} {text} {suffix}")


def _record(
    text: str,
    labels: str | list[str] | None,
    annotator: str,
    notes: str = "",
    template_key: str | None = None,
) -> dict:
    rec = {
        "id": str(uuid.uuid4()),
        "text": _clamp_text(text),
        "lang": "hu",
        "labels": _make_labels(labels),
        "source": "organic",
        "annotator": annotator,
        "notes": notes,
    }
    if template_key:
        rec["template_key"] = template_key
    return rec


def _unique_templates(texts: list[str]) -> list[str]:
    """Keep first occurrence of each normalized template sentence."""
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        key = text_key(text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def _unique_multi(
    items: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    seen: set[str] = set()
    unique: list[tuple[str, list[str]]] = []
    for text, cats in items:
        key = text_key(text)
        if key in seen:
            continue
        seen.add(key)
        unique.append((text, cats))
    return unique


def _partition(items: list, n_first: int, n_second: int, rng: random.Random) -> tuple[list, list]:
    """Disjoint partition of shuffled items into two lists of requested sizes."""
    pool = items.copy()
    rng.shuffle(pool)
    needed = n_first + n_second
    if len(pool) < needed:
        raise ValueError(
            f"Need {needed} unique templates, only {len(pool)} available"
        )
    return pool[:n_first], pool[n_first:needed]


def _wrapper_phrases() -> tuple[str, ...]:
    return tuple(
        _PREFIXES + _SUFFIXES + _LONG_INTROS + _LONG_MIDDLES + _LONG_CLOSINGS
    )


def _build_split(
    category_templates: dict[str, list[str]],
    negative_templates: list[str],
    multi_label_templates: list[tuple[str, list[str]]],
    long_single_templates: dict[str, list[str]],
    long_multi_templates: list[tuple[str, list[str]]],
    long_negative_templates: list[str],
    seed: int,
    diversify: bool,
    per_category: int,
    long_single_per_category: int,
) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []

    for category in CATEGORIES:
        templates = category_templates[category]
        if len(templates) < per_category:
            raise ValueError(
                f"Category {category} has only {len(templates)} train/test templates, "
                f"need {per_category}"
            )
        for text in templates[:per_category]:
            records.append(
                _record(
                    _diversify(text, rng, diversify),
                    category,
                    "seed-corpus-v1",
                    template_key=text_key(text),
                )
            )

        if long_single_per_category > 0:
            curated = list(long_single_templates.get(category, []))
            base_pool = templates[:per_category]
            long_texts: list[tuple[str, str]] = [
                (t, text_key(t)) for t in curated
            ]
            while len(long_texts) < long_single_per_category:
                core = rng.choice(base_pool)
                long_texts.append((_pad_to_long(core, rng), text_key(core)))
            for text, tkey in long_texts[:long_single_per_category]:
                records.append(
                    _record(
                        text, category, "seed-corpus-v3-long", "long-single", template_key=tkey
                    )
                )

    for text, cats in multi_label_templates:
        records.append(
            _record(
                _diversify(text, rng, diversify),
                cats,
                "seed-corpus-v2-multilabel",
                "multi-label",
                template_key=text_key(text),
            )
        )

    for text, cats in long_multi_templates:
        records.append(
            _record(
                text, cats, "seed-corpus-v3-long", "long-multi", template_key=text_key(text)
            )
        )

    for text in negative_templates:
        records.append(
            _record(
                _diversify(text, rng, diversify),
                None,
                "seed-corpus-v1",
                template_key=text_key(text),
            )
        )

    for text in long_negative_templates:
        records.append(
            _record(
                text, None, "seed-corpus-v3-long", "long-negative", template_key=text_key(text)
            )
        )

    rng.shuffle(records)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_by_category(records: list[dict]) -> dict[str, int]:
    counts = {cat: 0 for cat in CATEGORIES}
    counts["negative"] = 0
    counts["multi_label"] = 0
    counts["long"] = 0
    for record in records:
        labels = record["labels"]
        active = [cat for cat in CATEGORIES if labels[cat]]
        if len(record["text"]) >= 300:
            counts["long"] += 1
        if not active:
            counts["negative"] += 1
        else:
            if len(active) > 1:
                counts["multi_label"] += 1
            for cat in active:
                counts[cat] += 1
    return counts


def build_datasets(
    train_per_category: int = 22,
    test_per_category: int = 12,
    train_negatives: int = 40,
    test_negatives: int = 22,
    train_multi_label: int = 65,
    test_multi_label: int = 30,
    train_long_single_per_category: int = 8,
    test_long_single_per_category: int = 4,
    train_long_multi: int = 25,
    test_long_multi: int = 12,
    train_long_negative: int = 15,
    test_long_negative: int = 8,
    output_dir: str = "data/processed",
) -> None:
    assert set(_CATEGORY_TEXTS) == set(CATEGORIES), "_CATEGORY_TEXTS out of sync"
    assert_max_length()

    out = Path(output_dir)
    rng = random.Random(123)

    train_cat: dict[str, list[str]] = {}
    test_cat: dict[str, list[str]] = {}
    category_texts = {c: _unique_templates(_CATEGORY_TEXTS[c]) for c in CATEGORIES}
    min_unique = min(len(category_texts[c]) for c in CATEGORIES)
    if train_per_category + test_per_category > min_unique:
        raise ValueError(
            f"Requested {train_per_category}+{test_per_category} templates per category "
            f"but smallest category pool has only {min_unique} unique templates"
        )
    for category in CATEGORIES:
        tr, te = _partition(
            category_texts[category], train_per_category, test_per_category, rng
        )
        train_cat[category] = tr
        test_cat[category] = te

    train_neg, test_neg = _partition(_NEGATIVE_TEXTS, train_negatives, test_negatives, rng)
    train_multi, test_multi = _partition(
        _unique_multi(_MULTI_LABEL_EXAMPLES), train_multi_label, test_multi_label, rng
    )

    train_long_single: dict[str, list[str]] = {}
    test_long_single: dict[str, list[str]] = {}
    for category in CATEGORIES:
        curated = LONG_SINGLE.get(category, [])
        n_tr = min(len(curated), train_long_single_per_category)
        tr_cur, te_cur = _partition(
            curated,
            n_tr,
            min(len(curated) - n_tr, test_long_single_per_category),
            rng,
        ) if curated else ([], [])
        train_long_single[category] = tr_cur
        test_long_single[category] = te_cur

    long_multi_all = LONG_MULTI.copy()
    rng.shuffle(long_multi_all)
    long_multi_needed = train_long_multi + test_long_multi
    while len(long_multi_all) < long_multi_needed:
        text, cats = rng.choice(train_multi)
        long_multi_all.append((_pad_to_long(text, rng, min_len=380), cats))
    train_long_multi_list = long_multi_all[:train_long_multi]
    test_long_multi_list = long_multi_all[train_long_multi:long_multi_needed]

    long_neg_all = LONG_NEGATIVE.copy()
    rng.shuffle(long_neg_all)
    long_neg_needed = train_long_negative + test_long_negative
    while len(long_neg_all) < long_neg_needed:
        long_neg_all.append(_pad_to_long(rng.choice(train_neg), rng))
    train_long_neg_list = long_neg_all[:train_long_negative]
    test_long_neg_list = long_neg_all[train_long_negative:long_neg_needed]

    train_records = _build_split(
        category_templates=train_cat,
        negative_templates=train_neg,
        multi_label_templates=train_multi,
        long_single_templates=train_long_single,
        long_multi_templates=train_long_multi_list,
        long_negative_templates=train_long_neg_list,
        seed=42,
        diversify=True,
        per_category=train_per_category,
        long_single_per_category=train_long_single_per_category,
    )
    test_records = _build_split(
        category_templates=test_cat,
        negative_templates=test_neg,
        multi_label_templates=test_multi,
        long_single_templates=test_long_single,
        long_multi_templates=test_long_multi_list,
        long_negative_templates=test_long_neg_list,
        seed=99,
        diversify=False,
        per_category=test_per_category,
        long_single_per_category=test_long_single_per_category,
    )

    for split_name, recs in [("train", train_records), ("test", test_records)]:
        too_long = [r for r in recs if len(r["text"]) > MAX_TEXT_LENGTH]
        if too_long:
            raise ValueError(f"{split_name} has {len(too_long)} examples over {MAX_TEXT_LENGTH} chars")

    train_records, test_records, dedup_stats = split_disjoint(
        train_records, test_records, prefer="test"
    )

    template_overlap = find_template_overlap(train_records, test_records)
    dedup_stats["template_overlap_remaining"] = len(template_overlap)

    if find_overlap(train_records, test_records):
        raise RuntimeError("Exact train/test overlap remains after deduplication")
    if template_overlap:
        raise RuntimeError(
            f"Template-level train/test overlap remains ({len(template_overlap)})."
        )

    train_path = out / "train.jsonl"
    test_path = out / "test.jsonl"
    dedup_path = out / "dedup_stats.json"
    _write_jsonl(train_path, train_records)
    _write_jsonl(test_path, test_records)
    dedup_path.write_text(
        json.dumps(dedup_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_disjoint(train_path, test_path)
    test_path = out / "test.jsonl"
    dedup_path = out / "dedup_stats.json"
    _write_jsonl(train_path, train_records)
    _write_jsonl(test_path, test_records)
    dedup_path.write_text(
        json.dumps(dedup_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(train_records)} records to {train_path}")
    print(f"Wrote {len(test_records)} records to {test_path}")
    print(f"Dedup stats: {dedup_stats}")
    print("\nTrain split counts:")
    for key, count in sorted(_count_by_category(train_records).items()):
        print(f"  {key}: {count}")
    print("\nTest split counts:")
    for key, count in sorted(_count_by_category(test_records).items()):
        print(f"  {key}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Hungarian SPD datasets")
    parser.add_argument("--train-per-category", type=int, default=22)
    parser.add_argument("--test-per-category", type=int, default=12)
    parser.add_argument("--train-negatives", type=int, default=40)
    parser.add_argument("--test-negatives", type=int, default=22)
    parser.add_argument("--train-multi-label", type=int, default=65)
    parser.add_argument("--test-multi-label", type=int, default=30)
    parser.add_argument("--train-long-single-per-category", type=int, default=8)
    parser.add_argument("--test-long-single-per-category", type=int, default=4)
    parser.add_argument("--train-long-multi", type=int, default=25)
    parser.add_argument("--test-long-multi", type=int, default=12)
    parser.add_argument("--train-long-negative", type=int, default=15)
    parser.add_argument("--test-long-negative", type=int, default=8)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    build_datasets(
        train_per_category=args.train_per_category,
        test_per_category=args.test_per_category,
        train_negatives=args.train_negatives,
        test_negatives=args.test_negatives,
        train_multi_label=args.train_multi_label,
        test_multi_label=args.test_multi_label,
        train_long_single_per_category=args.train_long_single_per_category,
        test_long_single_per_category=args.test_long_single_per_category,
        train_long_multi=args.train_long_multi,
        test_long_multi=args.test_long_multi,
        train_long_negative=args.train_long_negative,
        test_long_negative=args.test_long_negative,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
