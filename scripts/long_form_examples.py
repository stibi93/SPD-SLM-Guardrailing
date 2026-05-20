"""Long-form Hungarian SPD examples (max 600 characters) for dataset building."""
from __future__ import annotations

MAX_TEXT_LENGTH = 600

# Single-label long examples keyed by category
LONG_SINGLE: dict[str, list[str]] = {
    "ethnicity": [
        (
            "Jó napot, már második napja próbálok ügyintézni a mobilbankban. Roma származású "
            "vagyok, és a helyi roma önkormányzati szervezet közösségi fejlesztési számlájára "
            "szeretnék utalni a havi támogatásomat. A fiókban azt mondták, online is megoldható, "
            "de nem találom a megfelelő kedvezményezettetípus-t. Tudnának segíteni, milyen "
            "dokumentumok kellenek a roma származás igazolásához?"
        ),
        (
            "Üdvözlöm a banki ügyfélszolgálatot! Német nemzetiségű magyar állampolgár vagyok, "
            "és a sváb egyesületünk éves adományát szeretném átutalni. A számlaszám megvan, "
            "de a közlemény rovatba mit írjak, ha nemzetiségi támogatásként szeretném "
            "feltüntetni? Korábban más banknál egyszerűbb volt, most elakadtam a netbankban."
        ),
        (
            "Segítséget kérek: szlovák származású vagyok, külföldön élő rokonoknak utalnék "
            "euróban. A tranzakció indításánál a rendszer kéri a nemzetiségi adatokat, "
            "mert korábban szlovák klub számlájára is utaltam. Nem értem, mi változott "
            "a felületen, és félek, hogy rossz adókat adok meg."
        ),
    ],
    "political_opinion": [
        (
            "Tisztelt Bank! DK-tag vagyok, és az éves tagdíjat szeretném befizetni a párt "
            "hivatalos számlájára. A mobilappban a kedvezményezett neve nem egyezik "
            "a honlapon látottal, ezért nem merem elküldeni az utalást. Kérem, erősítsék "
            "meg a helyes számlaszámot, mert politikai támogatásról van szó."
        ),
        (
            "Jó napot! Fidesz-szimpatizánsként eddig is utaltam a helyi szervezetnek, "
            "de most nagyobb összeget szeretnék adományozni kampánytámogatás címjén. "
            "A netbank jóváhagyásnál megakadtam, mert kétszer kérte ugyanazt az SMS-kódot. "
            "Konzervatív értékeimet nem szeretném félreértelmezni, csak gyors utalást kérek."
        ),
        (
            "Sürgős lenne: Momentum-tag vagyok, a havi tagdíjat és egy külön adományt "
            "is utalnék a párt számlájára. A fiókban azt mondták, online gyorsabb, "
            "de én baloldali meggyőződésem miatt szeretném, ha minden tétel külön "
            "szerepelne a kimutatáson. Milyen közleményt javasolnak?"
        ),
    ],
    "religion_belief": [
        (
            "Üdv! Katolikus vagyok, és az egyházam éves adománygyűjtésére szeretnék "
            "rendszeres utalást beállítani. A plébánia megadta a számlaszámot, "
            "de a csoportos beszedésnél nem találom a megfelelő kategóriát. "
            "Fontos lenne, hogy vallási adományként jelenjen meg, mert adóelőny miatt "
            "igazolást is kérek majd."
        ),
        (
            "Jó napot kívánok! Református hitet vallok, a gyülekezetünk templomfelújítására "
            "gyűjtünk. Szeretnék egyszeri nagyobb összeget utalni, de előtte biztosra "
            "szeretnék menni, hogy jó számlára megy. Korábban zsidó ismerősömnek is "
            "segítettem hasonló ügyben, nálam most a református közösség a címzett."
        ),
        (
            "Segítséget kérek: ateista világnézetet vallok, mégis humanista szervezetnek "
            "szeretnék adományozni. Nem vallási célú, de erkölcsi meggyőződésem része. "
            "A banki űrlapon kérnek 'támogatás jellegét', és nem tudom, mit írjak, "
            "hogy ne tévesztse össze vallási adománnyal."
        ),
    ],
    "trade_union": [
        (
            "Tisztelt Ügyfélszolgálat! Vasas szakszervezeti tag vagyok, és a havi tagdíj "
            "automatikus levonását szeretném beállítani. A munkahelyemen megváltozott "
            "a bankszámla, ezért a szakszervezeti számlára történő utalás elmaradt "
            "két hónapja. Kérem, segítsenek a csoportos beszedés újraindításában, "
            "mert a szakszervezeti tagságom függ tőle."
        ),
        (
            "Jó napot! Szakszervezeti tagdíjat szeretnék utalni, de a netbankban "
            "a kedvezményezett adatai nem menthetők el. Már harmadszor próbálkozom, "
            "telefonon sem sikerült. A szakszervezeti iroda sürget, mert lejár "
            "a tárgyalási jogosultságom. Milyen adatok hiányozhatnak?"
        ),
        (
            "Segítséget kérek sürgősen: szakszervezeti tag vagyok, és a sztrájkalap "
            "támogatását is szeretném utalni külön tételben. A fiókban azt mondták, "
            "hogy csak egy tranzakciót indíthatok egyszerre, de én két külön számlára "
            "kell utaljak tagdíjat és támogatást is."
        ),
    ],
    "genetic": [
        (
            "Üdvözlöm! Genetikai tesztem pozitív eredményt hozott BRCA1 mutációra, "
            "és a biztosítási kérelmemhez banki igazolást is kérek. Az orvosom "
            "írásos genetikai leletet adott, amit csatolni tudok. Örökletes "
            "rákhajlam miatt speciális hitelkonstrukciót keresek, ehhez kellene "
            "segítségük a dokumentumok benyújtásában."
        ),
        (
            "Jó napot! Családi genetikai szűrésen vettem részt, és kiderült, "
            "hogy örökletes szívbetegség kockázatom magas. A DNS-vizsgálat eredménye "
            "alapján halasztást kérek a hitel törlesztésére. A netbankban nem találom, "
            "hova tölthetem fel a genetikai jelentést."
        ),
        (
            "Segítséget kérek: genetikai tanácsadás után kaptam részletes leletet "
            "örökletes cukorbetegségről. A családban többen is érintettek, "
            "ezért közös számlát nyitnánk a feleségemmel, de előbb tudni szeretném, "
            "elfogadják-e genetikai dokumentumot fedezeti igazolásként."
        ),
    ],
    "health": [
        (
            "Jó napot! Cukorbeteg vagyok, inzulint használok, és hitelhalasztást "
            "szeretnék kérni a kezelésem miatt. Az orvosom írt igazolást, "
            "amit csatolni tudok, de a netbank űrlapja nem engedi feltölteni. "
            "Már két hete próbálkozom, a fiókban azt mondták, online gyorsabb lesz, "
            "de én elakadtam a dokumentum feltöltésénél."
        ),
        (
            "Üdv! Depresszióval kezeltetem magam, pszichiátriai kezelés alatt állok, "
            "és emiatt szeretném módosítani a hitel törlesztési ütemezésemet. "
            "Nem szívesen írok erről, de muszáj, mert egészségügyi okok miatt "
            "átmenetileg kevesebbet tudok fizetni. Kérem diszkrét ügyintézést."
        ),
        (
            "Segítséget kérek: látássérülésem miatt nem tudok normálisan "
            "elnavigálni a honlapon, pedig szívbeteg vagyok és sürgős "
            "ügyintézés lenne. A kártyámat tegnap nem sikerült letiltanom, "
            "most pedig orvosi igazolást is be kellene nyújtanom. Telefonos "
            "segítséget kérek, mert a felület nem akadálymentes."
        ),
    ],
    "sex_life_orientation": [
        (
            "Jó napot! Van egy lezbikus párom, és közös számlát szeretnénk nyitni. "
            "A netbank felületén nem egyértelmű, hogyan tudjuk ketten digitalisan "
            "aláírni a szerződést. Nem vagyunk házasok, de közös háztartásban "
            "élünk. Kérem, lépésről lépésre segítsenek, mert eddig csak "
            "magánszámlánk volt."
        ),
        (
            "Üdv! Meleg vagyok, és a párommal közös hitelt szeretnénk felvenni "
            "lakásvásárlásra. A banki tanácsadó azt mondta, online is elindítható, "
            "de a felületen nem találjuk a közös igénylés opciót. Fontos lenne, "
            "hogy mindketten szerepeljünk, mert hosszú kapcsolatban vagyunk."
        ),
        (
            "Segítséget kérek: biszexuális vagyok, transznemű partneremmel közös "
            "számlát nyitnánk. A névváltoztatás már megtörtént, de a rendszerben "
            "még a régi név szerepel. Az orientációnk és kapcsolati státuszunk "
            "miatt szeretnénk diszkrét, de gyors ügyintézést."
        ),
    ],
}

LONG_MULTI: list[tuple[str, list[str]]] = [
    (
        (
            "Nem sikerült tegnap letiltanom a kártyát. Van egy lezbikus párom, "
            "és szeretnénk majd közös számlát nyitni, viszont látássérülésem miatt "
            "nem tudok normálisan elnavigálni a honlapra, ahol ezt meg tudom tenni. "
            "Emellett cukorbeteg is vagyok, orvosi igazolást is tudok adni. "
            "Kérem, telefonos segítséggel intézzük, mert a felület nem akadálymentes!"
        ),
        ["sex_life_orientation", "health"],
    ),
    (
        (
            "Jó napot! Meleg vagyok, roma származású vagyok, és cukorbetegségem miatt "
            "közös számlát szeretnénk nyitni a párommal, miközben a roma közösségi "
            "szervezet számlájára is utalnék havi támogatást. A netbankban "
            "egyszerre sok adatot kérnek, és félek elrontani valamelyik űrlapot. "
            "Tudna valaki végigvezetni telefonon?"
        ),
        ["sex_life_orientation", "ethnicity", "health"],
    ),
    (
        (
            "Tisztelt Bank! Katolikus vagyok, szakszervezeti tag is, és szívbetegségem "
            "miatt halasztást kérek a hitel törlesztésére. Szeretnék egyrészt tagdíjat "
            "utalni a szakszervezetnek, másrészt adományt az egyháznak, harmadrészt "
            "pedig orvosi igazolást benyújtani. A mobilappban nem férnek el "
            "egyszerre ennyi kérelem."
        ),
        ["religion_belief", "trade_union", "health"],
    ),
    (
        (
            "Segítséget kérek! DK-tag vagyok, depresszióval kezeltetem magam, "
            "és emiatt nem tudtam időben befizetni a párt tagdíját. Politikai "
            "nézeteim miatt fontos lenne rendezni a tartozást, de egészségügyi "
            "okokból halasztást is kérek a magánhitelre. Milyen dokumentumok "
            "kellenek mindkét ügyhöz?"
        ),
        ["political_opinion", "health"],
    ),
    (
        (
            "Üdv! Örökletes rák szindrómám van, genetikai teszttel igazolt, "
            "és jelenleg kemoterápiás kezelés alatt állok. Emellett leszbikus "
            "párommal közös számlát nyitnánk, hogy a kórházi kiadásokat "
            "együtt kezeljük. A netbank nem engedi feltölteni egyszerre "
            "a genetikai és az orvosi papírt."
        ),
        ["genetic", "health", "sex_life_orientation"],
    ),
    (
        (
            "Jó napot! Német nemzetiségű vagyok, református hitet vallok, "
            "és mindkét közösség számlájára szeretnék utalni. A nemzetiségi "
            "egyesület és a templom külön számlát adott, de én egy tranzakcióban "
            "szeretném megoldani, ha lehetséges. Korábban egyszerűbb volt "
            "a netbankban, most elvesztem a menüben."
        ),
        ["ethnicity", "religion_belief"],
    ),
    (
        (
            "Sürgős ügy! Szakszervezeti tag vagyok, Fidesz-szimpatizáns is, "
            "és mindkét szervezetnek utalnom kellene ma. Emellett asztmás vagyok, "
            "most rosszabbodott az állapotom, ezért orvosi igazolást is "
            "szeretnék csatolni halasztási kérelemhez. Kérem, segítsenek "
            "priorizálni, mert egyszerre több határidőm is lejár."
        ),
        ["trade_union", "political_opinion", "health"],
    ),
    (
        (
            "Üdvözlöm! Biszexuális vagyok, zsidó hitet vallok, és hallássérülésem "
            "miatt jelnyelvi tolmácsos ügyintézést kérek. A párommal közös "
            "számlát nyitnánk, miközben a hitközségnek is adományoznánk. "
            "A videós azonosítás nem működött nálam, ezért személyes "
            "időpontot szeretnék akadálymentes fiókban."
        ),
        ["sex_life_orientation", "religion_belief", "health"],
    ),
    (
        (
            "Jó napot! BRCA2 mutációm van, magas vérnyomással kezelnek, "
            "és emiatt biztosítási és hitelügyet is egyszerre kellene intéznem. "
            "A genetikai leletet már megkaptam, a kardiológiai igazolást "
            "holnap kapom. A netbank csak egy feltöltést enged, pedig mindkettő "
            "kellene a kérelemhez."
        ),
        ["genetic", "health"],
    ),
    (
        (
            "Segítséget kérek! Transznemű vagyok, Momentum-támogató is, "
            "és névváltoztatás után frissíteni kellene a bankszámlámat. "
            "Emellett epilepsziás vagyok, orvosi igazolást is csatolok "
            "a limitemódosítási kérelemhez. A fiókban hosszú sort "
            "álltam, online gyorsabb lenne, ha működne."
        ),
        ["sex_life_orientation", "political_opinion", "health"],
    ),
]

LONG_NEGATIVE: list[str] = [
    (
        "Jó napot! Már harmadik napja próbálok átutalást indítani a netbankban, "
        "de mindig megakad a jóváhagyásnál. A telefonra nem jön SMS-kód, "
        "a fiókban pedig azt mondták, frissítsem az appot. Eközben a bérleti "
        "díj határideje holnap lejár, és aggódom, hogy késedelmes lesz. "
        "Nem vagyok technikai ember, kérem, lépésről lépésre segítsenek, "
        "hogyan tudom biztonságosan elküldeni a megbízást anélkül, "
        "hogy újra elölről kelljen kezdenem."
    ),
    (
        "Tisztelt Ügyfélszolgálat! A céges számlánkra tegnap beérkezett összeg "
        "nem jelenik meg a mobilbankban, pedig a partner már elküldte az "
        "igazolást. A könyvelő sürget, mert ma kell kifizetnünk a beszállítót. "
        "Ellenőrizték már a tranzakció azonosítót, az rendben van. "
        "Kérem, nézzék meg, miért csúszik a jóváírás, és tudnak-e "
        "ideiglenes igazolást adni a partner felé."
    ),
]


def assert_max_length() -> None:
    for cat, texts in LONG_SINGLE.items():
        for text in texts:
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(f"LONG_SINGLE[{cat}] exceeds {MAX_TEXT_LENGTH}: {len(text)}")
    for text, _ in LONG_MULTI:
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"LONG_MULTI exceeds {MAX_TEXT_LENGTH}: {len(text)}")
    for text in LONG_NEGATIVE:
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"LONG_NEGATIVE exceeds {MAX_TEXT_LENGTH}: {len(text)}")
