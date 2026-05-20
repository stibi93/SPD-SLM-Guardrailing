"""
Realistic Hungarian retail-banking chat scenarios (no Article 9 data in negatives).
Used to expand training data — templates must stay unique for train/test splits.
"""
from __future__ import annotations

# --- Negatives: everyday problems customers ask in chat (no SPD) ---

BANKING_NEGATIVE_TEXTS: list[str] = [
    # Cards & payments
    "A kártyámat elfogta az ATM, mit tegyek most?",
    "Nem tudok fizetni a boltban, contactless nem működik.",
    "A Google Pay nem adja hozzá a kártyámat, hibakódot kapok.",
    "Kétszer terheltek le ugyanarról a vásárlásról, visszatérítést kérek.",
    "Elutasították a kártyás fizetésemet külföldön, miért?",
    "A hitelkártya PIN-kódomat zárolták három rossz próba után.",
    "Szeretném a kártyát ideiglenesen befagyasztani, elvesztettem a tárcámat.",
    "A virtuális kártyám lejárt, hogyan kapok újat az appban?",
    "Miért kell minden online vásárlásnál újra jóváhagynom az appban?",
    "A POS terminál nem fogadta el a kártyámat, másik számláról próbáljam?",
    # Transfers & beneficiaries
    "Tegnapi utalásom még mindig feldolgozás alatt van, normális ez?",
    "Rossz számlaszámra utaltam, vissza lehet vonni?",
    "A kedvezményezett neve nem egyezik a számlával, miért utasítják el?",
    "SEPA utalás 24 óra után sem érkezett meg a címzettnek.",
    "Hogyan állíthatok be rendszeres havi utalást a bérleti díjra?",
    "A közlemény rovatba mit írjak, ha számlaszámot küldök a címzettnek?",
    "Külföldi deviza utalásnál milyen költségek merülnek fel?",
    "A gyorsutalás limitje mennyi, és hogyan emelhető?",
    "Nem találom a mentett kedvezményezetteket az új netbank felületen.",
    "A csoportos beszedés megbízásomat szeretném módosítani.",
    # App & digital banking
    "Nem tudok belépni a mobilbankba, elfelejtettem a jelszót.",
    "Az app összeomlik bejelentkezés után, Android 14-en.",
    "A QR-kódos bejelentkezés nem működik, kamerát nem engedélyezi.",
    "Frissítettem az appot, és eltűnt a számláim listája.",
    "Push értesítések nem jönnek, hol kapcsolom be?",
    "A netbankban nem látom a folyószámla részleteit, csak az összeget.",
    "Két faktoros azonosítás SMS-e késik, nem tudok utalni.",
    "Biometrikus belépés hibát jelez, vissza PIN-re szeretnék állni.",
    "A tranzakció megerősítő push nem érkezik meg.",
    "Hol tudom letölteni a mobilbank használati útmutatóját magyarul?",
    # Accounts & packages
    "Szeretnék Premium számlacsomagra váltani, mi jár hozzá?",
    "Melyik számlacsomag ingyenes fiataloknak 25 év alatt?",
    "Hogyan zárhatom le a régi folyószámlámat, ha már nem használom?",
    "Közös számlát szeretnénk nyitni a párommal, milyen dokumentum kell?",
    "Gyermek számlát nyitnék 14 évesnek, szülői hozzájárulás kell?",
    "Devizaszámlát szeretnék USD-ben, milyen a számlavezetési díj?",
    "A számlanyitás online elakadt a videós azonosításnál.",
    "Miért számít fel havi számlavezetési díjat, ha ingyenes csomagom van?",
    "Szeretném összevonni a régi és az új számlámat egy appban.",
    "A megtakarítási számlám kamata miért csökkent ebben a hónapban?",
    # Loans & mortgages
    "Mennyi a fennmaradó hiteltőkém a lakáshitelnél?",
    "Szeretném előtörleszteni a személyi kölcsönöm egy részét.",
    "A hiteligazolást mikor kapom meg e-mailben?",
    "A kamatfixálás lejár, milyen opcióim vannak most?",
    "Jelzálog törlését kérem, mert kifizettem a hitelt.",
    "A babaváró hitel törlesztése automatikusan megy tovább?",
    "Mennyi a THM a folyószámlahitelnél jelenleg?",
    "Hitelkalkulátor nem tölt be, szeretnék személyi kölcsönt.",
    "A törlesztési napot szeretném 15-ről 20-ra módosítani.",
    "Mi történik, ha késedelmes lesz a hiteltörlesztés?",
    # Fees, disputes, fraud
    "Ismeretlen tétel jelent meg a kivonaton, panaszt szeretnék tenni.",
    "Visszaterhelést kérek, mert nem kaptam meg a rendelést.",
    "Gyanús e-mailt kaptam a bank nevében, hova küldjem?",
    "A banki díj visszatérítését szeretném kérni, jogosulatlan levonás volt.",
    "Chargeback folyamatát szeretném elindítani kártyás vásárlásnál.",
    "Valaki ismeretlenül regisztrálta a kártyámat egy webshopban.",
    "A csalás jelentés után mikor kapok visszajelzést?",
    "Kérem blokkolják a kártyát, gyanús tranzakciót látok.",
    "A biztonsági riasztás SMS valódi, vagy phishing?",
    "Hogyan állíthatok be napi limitet az internetes vásárlásra?",
    # Documents & tax
    "Kérem az éves kamatjóváírás igazolást PDF-ben.",
    "Bankszámla igazolást kérek a lakáshitelhez, sürgős.",
    "Hol tudom letölteni a 2025-ös tranzakciós kimutatást?",
    "A NAV részére kérek igazolást a számlaegyenlegről.",
    "Külföldi munkáltatónak kell angol nyelvű banki referencialevél.",
    "A záradékos igazolást hogyan kérem online?",
    "A számlanyitási szerződést nem találom az archívumban.",
    "Kérem a hitel törlesztési igazolást az elmúlt 12 hónapról.",
    "A céges számla aláírási címpéldányát hol töltsem fel?",
    "Adóelőleg befizetését szeretném csoportos beszedéssel intézni.",
    # Branch & service
    "Szeretnék időpontot foglalni számlanyitásra a belvárosi fiókba.",
    "Melyik fiókban van sorbanállásmentes ügyintézés?",
    "A telefonos ügyfélszolgálat nem vette fel, chaten tudnak segíteni?",
    "Ügyintézőt kérek, mert bonyolult a devizaügy.",
    "Hol tudok érmebefizetést csinálni a fiókban?",
    "A fiókban kért dokumentumot e-mailben is megkapom?",
    "Sürgős lenne: hol van legközelebbi ATM készpénzfelvételhez?",
    "A bankfüzetben látott kamat nem egyezik az appban láthatóval.",
    "Videós ügyintézéshez milyen okmányt tartsak kéznél?",
    "A chatbot nem érti a kérdésemet, emberi ügyintézőt kérek.",
    # Business / corporate (still retail tone)
    "Céges számlán nem látszik a tegnapi bejövő számla.",
    "POS terminál kártyás kifizetése nem ír jóvá a számlán.",
    "A vállalkozói számlacsomag díját szeretném csökkenteni.",
    "Kérek több aláírót a céges netbankhoz.",
    "A könyvelő nem fér hozzá a kivonathoz, jogosultságot adok neki?",
    "ÁFA-s számla befizetését szeretném ütemezni.",
    "Céges hitelkeretem felhasználtságát hol látom?",
    "A számlázó program nem kapcsolódik a bankhoz, mit tegyek?",
    "Külföldi partnernek EUR számláról szeretnék számlát küldeni.",
    "A céges kártya napi limitjét hol emelhetem?",
    # Life events (non-SPD)
    "Öröklés miatt nagyobb összeg érkezett, adózni kell?",
    "Eladom a lakást, a vevői letétet hogyan intézzük a bankon keresztül?",
    "Esküvőre gyűjtünk, közös cél számlát nyitnánk.",
    "Költözés miatt megváltozott a lakcím, hol jelzem?",
    "Diákhitel befizetését szeretném automatikussá tenni.",
    "Nyugdíjasként kedvezményes számlacsomagot keresek.",
    "Külföldre költözöm, mi lesz a bankszámlámmal?",
    "Munkanélküli juttatás érkezik erre a számlára, rendben van?",
    "Álláskereső vagyok, számlanyitás ingyenes lehet?",
    "Megszűnt a munkaviszonyom, hiteltörlesztés továbbra is megy?",
    # Frustration / edge UX (still no SPD)
    "Harmadjára hívom ugyanazért, nem kaptam megoldást.",
    "A chatben ígért visszahívás nem érkezett meg.",
    "Az ügyintézés két hete tart, hol tart a folyamat?",
    "A rendszer azt írja, technikai hiba, mikor lesz javítva?",
    "Nem értem a magyarázatot, egyszerűbben el tudnák mondani?",
    "Este 10 után is tudok telefonon ügyet intézni?",
    "A várakozási idő túl hosszú, van prioritásos sáv?",
    "Korábban másképp működött a netbank, hol a régi menü?",
    "A szerződésemben más díj szerepel, mint amit levontak.",
    "Kérem, hogy az ügyemet egy helyen követhessük, mert sok ügyszámom van.",
]

BANKING_NEGATIVE_LONG: list[str] = [
    (
        "Jó napot, tegnap este próbáltam 450 ezer forintot utalni a felújító vállalkozónak, "
        "a tranzakció 'feldolgozás alatt' maradt, a vállalkozó nem kapta meg. A mobilapp "
        "nem enged új megbízást indítani, mert azt írja, van függőben lévő művelet. "
        "Telefonon 40 percet vártam, megszakadt a hívás. Kérem, nézzék meg a tranzakció "
        "státuszát, és ha kell, indítsák újra vagy mondják meg, mikor érkezik meg."
    ),
    (
        "Tisztelt Bank, céges ügyfélként a könyvelőnk nem éri el a netbankot, holnap "
        "ÁFA-bevallás van. A felhasználói jogosultságokat módosítottuk, de továbbra is "
        "hibaüzenet jön. Emellett a tegnapi POS bevételek egy része nem jelent meg a "
        "számlán. Sürgősen kérek technikai ellenőrzést és ideiglenes limit emelést a "
        "kimenő utalásokra, hogy a adót tudjuk befizetni."
    ),
    (
        "Segítséget kérek: külföldi utazás közben minden kártyás fizetésem elutasításra "
        "került, pedig van elég pénz a számlán. A mobilbankban engedélyeztem a külföldi "
        "használatot, de a boltban mégsem működött. Készpénzfelvétel sem sikerült az ATM-en. "
        "Hogyan tudom azonnal feloldani a blokkolást, vagy kérek segélykártyát?"
    ),
    (
        "Üdv, lakáshitel kamatfixálásom jár le két hét múlva, és nem értem a bank által "
        "küldött levélben a három opciót. Szeretnék időpontot vagy telefonos tanácsadást, "
        "hogy ne emelkedjen túl sokat a törlesztőrészlet. A netbank kalkulátora más számot "
        "mutat, mint amit az ügyintéző mondott múlt héten."
    ),
    (
        "Elvesztettem a tárcámat metróban, a bankkártyát azonnal le szeretném tiltani. "
        "A mobilappban a tiltás gomb nem reagál, csak forog. Van másik telefonszám, "
        "ahol 24 órában elérik a bank? Később új kártyát is kérek, de most a fő, "
        "hogy senki ne tudjon levonni."
    ),
]

# Extra SPD-positive templates in realistic banking-chat framing (unique sentences)
BANKING_SPD_EXTRA: dict[str, list[str]] = {
    "health": [
        "A munkáltatóm kéri a táppénz igazolást — krónikus betegségemről tudnak banki levelet adni?",
        "Orvosi kezelésem miatt átmenetileg csökkent a jövedelmem, hitelhalasztást kérek.",
        "A biztosító kérte a kórházi kezelés igazolását, feltölthetem a netbank ügyfélhelyére?",
        "Gyógyszerköltségemről kérek részletfizetést a hitelkártyán, mert elfogyott a keret.",
        "Szívultrahang eredménye alapján módosítani kell a biztosítási nyilatkozatomat a banknál.",
        "Táppénz idejére felfüggeszthető a személyi kölcsön törlesztése?",
        "Allergiás reakció miatt sürgős orvosi költséget szeretnék előlegezni a számláról.",
        "Rehabilitációs kezelésem díját részletekben szeretném fizetni a banktól kapott hitelből.",
        "A fogyatékosságom miatt speciális ügyintézést kérek a hitelügyben.",
        "Műtéti utóm miatt nem tudok dolgozni, mikor kezdődik újra a törlesztés?",
    ],
    "sex_life_orientation": [
        "Meleg párommal közös hitelt vennénk, de a netbank csak egy fő adósa számára engedélyez.",
        "Leszbikus vagyok, a banki űrlapon a családi státusz rovat zavarba hoz.",
        "Közös számlát nyitnánk bejegyzett élettárssal, milyen papírok kellenek?",
        "Transznemű vagyok, a személyazonosítóm már frissült, a banki adatok nem.",
        "A párom és én melegok vagyunk, kedvezményes családi csomag jár-e nekünk?",
        "Biszexuális vagyok, de ez nem releváns a hitelnél — mégis kitöltöm?",
        "Az LMBTQ közösség bankszámlájára adományoznék, megadhatom a kedvezményezettet?",
        "Meleg párommal lakáshitelt vennénk, közös jövedelmet kell igazolni?",
        "Nem akarom, hogy a chatben rögzítsék a szexuális irányultságomat, de szükséges az ügyhöz.",
        "Regisztrált élettárssal szeretnénk közös biztosítást kötni a hitelhez.",
    ],
    "ethnicity": [
        "Roma származású vállalkozóként mikrohitelhez kérek tájékoztatást a chaten.",
        "Nemzetiségi önkormányzati számlára utalnék — milyen közleményt írjak?",
        "Szlovák rokonoknak utalok, a nemzetiségi adatokat kéri a rendszer.",
        "A roma közösségi program számlaszámát hol rögzítsem kedvezményezettként?",
        "Német nemzetiségűként pályázati támogatást utalok, adóigazolás kell?",
    ],
    "political_opinion": [
        "Párt tagdíját szeretném utalni, de a kedvezményezett neve nem található.",
        "Kampányadományt szeretnék küldeni, van limit az utalásra?",
        "Politikai szervezet számlájára megy az utalás, közleményt kérek mintát.",
        "Helyi párt szervezetének adományoznék, biztosítani tudják a címet?",
    ],
    "religion_belief": [
        "Templom felújítására adományoznék, egyházi számlaszámot hogyan ellenőrizzek?",
        "Református gyülekezetnek szeretnék rendszeres utalást beállítani.",
        "Zsinagóga adomány igazolását kérem a banktól az adóbevalláshoz.",
        "Ateista vagyok, de humanista alapítványnak utalok — milyen kategóriát válasszak?",
    ],
    "trade_union": [
        "Szakszervezeti tagdíj csoportos beszedését szeretném újraindítani új számláról.",
        "A szakszervezet másik számlaszámot adott, hogyan módosítom a megbízást?",
        "Tagdíj utalás után kérek bizonylatot a munkáltatónak.",
    ],
    "genetic": [
        "Genetikai teszt eredményét csatolom a biztosítási kérelemhez, hol tölthetem fel?",
        "Örökletes betegség kockázata miatt kérek hitelbiztosítási felmérést.",
        "BRCA vizsgálat alapján módosul a kedvezményem, mit küldjek be?",
    ],
}

BANKING_MULTI_EXTRA: list[tuple[str, list[str]]] = [
    (
        "A kártyám tegnap nem működött, közben cukorbeteg vagyok és sürgős gyógyszert kell vennem — "
        "tudnak segíteni limit emeléssel?",
        ["health"],
    ),
    (
        "Utalást szeretnék indítani a szakszervezetnek, de a netbank összeomlik; "
        "közben DK-tag vagyok, a párt számlájára is kellene mennie ma.",
        ["trade_union", "political_opinion"],
    ),
    (
        "Lakáshitelhez kérek halasztást: szívproblémám van, és a párommal melegok vagyunk, "
        "közös törlesztők vagyunk.",
        ["health", "sex_life_orientation"],
    ),
    (
        "Katolikus vagyok, templomnak utalnék, de rossz számlaszámot adtam meg tegnap — "
        "visszavonható?",
        ["religion_belief"],
    ),
    (
        "Roma származású vagyok, közösségi hitelhez kérek tanácsot; a netbank nem érti "
        "a kedvezményezett típusát.",
        ["ethnicity"],
    ),
]
