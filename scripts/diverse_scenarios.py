"""
Diverse Hungarian banking-chat examples designed to cover linguistic patterns
not yet represented: negation, indirect/3rd-person disclosure, mixed-topic
sentences, frustration tone, formal letter style, question form, multi-sentence
context, implicit vs explicit SPD mentions.

All sentences are unique (verified at build time) and max 600 characters.
"""
from __future__ import annotations

# ── Negation variants (still mentions the category → still positive label) ──
# "nem vagyok X" — project annotation decision: still counts, as the category
# is named and the model should detect the *mention* for guardrailing.
NEGATION_SPD: dict[str, list[str]] = {
    "health": [
        "Nem vagyok cukorbeteg, de az anyukám igen és az ő nevére kérnénk hitelt.",
        "Ugyan nem szedem gyógyszert most, de mostanában diagnosztizáltak magas vérnyomással.",
        "Nem beteg vagyok, csak a rendszeres kardiológiai ellenőrzőm eredményeit csatolom.",
        "Nem depressziós időszak van, de a pszichiáter ajánlólevelét kérték a biztosítótól.",
        "Nem kezeltetem magam, de genetikai kockázatom van, emiatt kérek fedezetet.",
    ],
    "sex_life_orientation": [
        "Nem akarom részletezni, de a párom azonos nemű, közös számlát nyitnánk.",
        "Nem kérdés az orientációm, de az LMBTQ-szervezetnek szeretnék utalni.",
        "Nem a magánéletemről van szó, csak a regisztrált élettársammal nyitnánk számlát.",
    ],
    "religion_belief": [
        "Nem vagyok mélyen vallásos, de a plébánia számlájára rendszeres adományt utalnék.",
        "Nem tartom fontosnak a vallási hovatartozásomat, mégis a mecset alapítványnak adományoznék.",
    ],
    "ethnicity": [
        "Nem határozom meg magam etnikailag, de a roma önkormányzatnak utalnék.",
        "Nem fontosabb a származásom, csak a sváb egyesület számlájára küldöm az adományt.",
    ],
    "political_opinion": [
        "Nem vagyok aktív politikus, de DK-tagdíjat kell fizessek.",
        "Nem politizálok nyilvánosan, de a Momentum helyi szervezetének adományoznék.",
    ],
    "trade_union": [
        "Nem sztrájkolok, de a szakszervezeti tagdíjamat be kell fizetnem.",
        "Nem aktív szakszervezeti tisztviselő vagyok, csak tagdíjat utalnék.",
    ],
    "genetic": [
        "Nem beteg vagyok, de genetikai tesztem pozitív lett, ezért kérek fedezetet.",
        "Nem öröklöm el biztosan, de BRCA-szűrésem pozitív lett.",
    ],
}

# ── 3rd-person / family member mention ──
THIRD_PERSON_SPD: dict[str, list[str]] = {
    "health": [
        "Az apám cukorbeteg, az ő hitelének halasztását kérem meghatalmazással.",
        "A lányom epilepsziás, az ő bankszámlájának kezeléséhez kérek hozzáférést.",
        "A férjem szívinfarktust kapott, a közös hitelünkhöz törlesztési moratóriumot kérnénk.",
        "Az anyósomnak rákos megbetegedése van, ő kért meg, hogy intézzem a bankügyet.",
        "A bátyám veseelégtelenségben szenved, dialízis miatt kér törlesztési halasztást.",
        "A gyerekemet autizmussal diagnosztizálták, speciális megtakarítási számlát nyitnék.",
        "Az unokám Down-szindrómás, az ő gyámszámlájához kell hozzáférés.",
        "A szüleim koronavírus után maradványtünetekkel küzdenek, ők kértek meg segíteni.",
    ],
    "ethnicity": [
        "A férjem roma származású, a közösségi hitelüket együtt intéznénk.",
        "A szomszédom, aki cseh–magyar kettős állampolgár, szeretne ideköltözni és számlát nyitni.",
        "Az apám görög bevándorló volt, örökségét magyarban kell kezelnünk.",
    ],
    "sex_life_orientation": [
        "A fiam meleg, a párjával lakáshitelt vennének, segítség kellene a közös igényléshez.",
        "A testvérem transznemű, az adatait frissíteni kell a bankban a névváltoztatás után.",
        "A barátnőm leszbikus, párjával közös számlát nyitnának, de nem tudják hogyan.",
    ],
    "religion_belief": [
        "Az édesanyám buzgó katolikus, az egyházi adományát intézem helyette.",
        "A férjem ortodox zsidó, a hitközség számlájára kell utalnunk.",
        "A szomszédaim muszlimok, nekik adományozunk az Eid alkalmából.",
    ],
    "political_opinion": [
        "A munkáltatóm Fidesz-közeli szervezet, a párt számlájára kell utalni.",
        "A cégünk MSZP-s alapítványt támogat, a tagdíjat mi folyósítjuk.",
    ],
    "trade_union": [
        "A munkatársaim szakszervezeti tagok, a tagdíjat én gyűjtöm össze és utalom.",
        "A kollégám nevében a szakszervezeti tagdíjat szeretném befizetni.",
    ],
    "genetic": [
        "Az anyám BRCA-pozitív, a biztosítási kérelmét intézem helyette.",
        "A testvéremnek örökletes neuropátiája van, ő küldött meghatalmazást.",
    ],
}

# ── Indirect / implicit language (no explicit category keyword) ──
IMPLICIT_SPD: dict[str, list[str]] = {
    "health": [
        "Tartós kezelésem miatt a havi kiadásaim megugrottak, hitelhalasztást kérek.",
        "Az orvosom javaslatára módosítanom kell a biztosítási nyilatkozatom.",
        "Egy hosszabb kórházi tartózkodás után próbálom rendezni a pénzügyeimet.",
        "Rendszeres gyógyszerköltségeim vannak, ezért kérek részletfizetési lehetőséget.",
        "A kezelésem miatt csökkent a munkaidőm, a jövedelmem is visszaesett.",
        "Egészségügyi okok miatt nem tudtam dolgozni három hónapig.",
        "Az utóbbi időben sokat kellett az orvosra költeni, a törlesztő nehéz lett.",
        "Krónikus állapotom kezelési díja megemelte a havi kiadásaimat.",
    ],
    "religion_belief": [
        "Az ünnepünk előtt egy nappal szeretném az adományt elutalni.",
        "A közösségünk számlájára havi rendszeres utalást állítanék be.",
        "Erkölcsi meggyőződésem alapján adományozok egy civil szervezetnek.",
    ],
    "sex_life_orientation": [
        "A partnerem és én szeretnénk hivatalossá tenni közös pénzügyeinket.",
        "Ketten vagyunk és közös számlát nyitnánk, de a rendszer csak egy főt enged.",
        "A párom neve a bankban még nem szerepel, frissíteni kellene a meghatalmazást.",
    ],
    "political_opinion": [
        "Egy civil szervezetnek szeretnék utalni, amely politikai tevékenységet végez.",
        "A választási kampányhoz szeretnék adományozni az egyik jelölt szervezetének.",
    ],
    "ethnicity": [
        "A közösségünk számlájára utalnék, amelynek kisebbségi státusza van.",
        "Rokonaimnak utalnék külföldre, az ottani kisebbségi szervezeten keresztül.",
    ],
    "trade_union": [
        "Munkavállalói szervezetünk számlájára kell havi rendszeres befizetést indítani.",
        "A kollektív megállapodásban rögzített hozzájárulást kell utalni a szervezetnek.",
    ],
    "genetic": [
        "Örökletesen megnövelt kockázatom van egy betegségre, ezért kérek speciális feltételeket.",
        "Egy genetikai vizsgálat eredménye miatt kell a biztosítási nyilatkozatom módosítani.",
    ],
}

# ── Frustration / urgency tone SPD messages ──
FRUSTRATED_SPD: dict[str, list[str]] = {
    "health": [
        "Már harmadik hete próbálom elintézni a hitelhalasztást, cukorbetegségem igazolása megvan!",
        "Sürgős! A kórházi kezelésem ma kezdődik, a törlesztőt nem tudom fizetni, kérem segítsenek!",
        "A betegségem miatt nem tudtam dolgozni, és már késedelmes vagyok — miért nem intézik?",
        "Orvosi papírt is csatoltam, mégis elutasítják a halasztási kérelmemet!",
    ],
    "sex_life_orientation": [
        "Már hetedszer próbáljuk elindítani a közös számlát a párommal — melegek vagyunk, ez probléma?",
        "Miért nem tudunk közös hitelt igényelni, ha bejegyzett élettársak vagyunk?",
    ],
    "religion_belief": [
        "Az egyházi adományom határideje ma van, miért nem megy az utalás?",
        "Református vagyok, és a templomnak járó adományt nem engedik el — miért?",
    ],
    "trade_union": [
        "A szakszervezeti tagdíj határideje ma, és a netbank nem hagy utalást indítani!",
    ],
    "political_opinion": [
        "A párttagdíj fizetési határidő lejárt, de a rendszer hibát jelez — azonnal segítség kell!",
    ],
    "genetic": [
        "Genetikai leletemet három hete feltöltöttem, a biztosítási kérelmem mégis feldolgozás alatt.",
    ],
    "ethnicity": [
        "Roma vagyok, és már negyedik alkalommal utasítják el az igényemet — ez diszkrimináció?",
    ],
}

# ── Formal / letter-style SPD (official, polite Hungarian) ──
FORMAL_SPD: dict[str, list[str]] = {
    "health": [
        "Tisztelettel tájékoztatom Önöket, hogy krónikus betegségem miatt kérem törlesztési haladék biztosítását.",
        "Hivatkozással az orvosi dokumentációra, kérem a kölcsönszerződés módosítását egészségügyi okok miatt.",
        "Alulírott, mint a bank ügyfele, ezúton kérem halasztás megadását az egészségi állapotomra tekintettel.",
    ],
    "sex_life_orientation": [
        "Alulírott és bejegyzett élettársam kérjük a közös bankszámla megnyitásának engedélyezését.",
        "Ezúton jelzem, hogy élettársam azonos nemű, a közös hitelkérelmet ennek megfelelően tessék kezelni.",
    ],
    "religion_belief": [
        "Alulírott, mint római katolikus hívő, kérem az egyházi adományok csoportos megbízásának beállítását.",
        "Hivatkozással a vallásszabadságra, kérem a vallási szervezet számlájára való rendszeres utalás engedélyezését.",
    ],
    "ethnicity": [
        "Alulírott, mint magyarországi roma vállalkozó, kérem a mikrohitel igénylésének elbírálását.",
        "Hivatkozással a kisebbségi támogatási programra, kérem a pályázati összeg utalásának feldolgozását.",
    ],
    "political_opinion": [
        "Alulírott, mint a demokratikus ellenzék tagja, kérem a pártadományozás technikai feltételeit.",
        "Ezúton jelzem párttagságomat, és kérem a tagdíj-átutalás lehetőségét bankszámlamról.",
    ],
    "trade_union": [
        "Alulírott szakszervezeti tag kérje a havi tagdíj automatikus utalásának beállítását.",
    ],
    "genetic": [
        "Hivatkozással a genetikai tanácsadás eredményére, kérem a biztosítási feltételek felülvizsgálatát.",
    ],
}

# ── Question-form SPD (user asking the bank a question that reveals SPD) ──
QUESTION_SPD: dict[str, list[str]] = {
    "health": [
        "Orvosi igazolással kérhetek-e törlesztési szünetet, ha krónikus betegségem van?",
        "Van-e speciális hitelprogramjuk mozgáskorlátozott ügyfelek részére?",
        "Alkalmas-e a mobilbank vakok számára, ha látáskárosodásom van?",
        "Rákos kezelés idejére felfüggeszthető-e a hitelkártya törlesztési kötelezettség?",
        "Milyen dokumentum kell a cukorbetegség miatti halasztás igazolásához?",
        "Pszichiátriai kezelés alatt állok — van-e erre vonatkozó kedvezményes program?",
        "Kapok-e könnyítést, ha a munkaképtelenségem tartós és orvosi igazolt?",
    ],
    "sex_life_orientation": [
        "Bejegyzett élettársak igényelhetnek-e közös lakáshitelt ugyanolyan feltételekkel?",
        "Hogyan módosíthatjuk a bankszámla adatait, ha a párom transznemű és nevet változtatott?",
        "Az LMBTQ civil szervezet számlája felvehető-e kedvezményezettként?",
        "Melegek vagyunk — egyforma-e az elbírálás a közös hitelnél?",
    ],
    "religion_belief": [
        "Egyházi közösség számlájára indíthatok-e rendszeres megbízást a netbankban?",
        "A vallási szervezetnek adott adomány levonható-e az adóból, és kell-e bankigazolás?",
        "Milyen közleményt javasol, ha iszlám alapítványnak utalok adományt?",
    ],
    "ethnicity": [
        "Roma vállalkozóként igénybe vehetem-e a kedvezményes KKV hitelprogramot?",
        "Kisebbségi önkormányzati számlára is küldhetek-e rendszeres megbízást?",
        "Külföldi rokonoknak utalnék egy másik ország kisebbségi szervezetének — hogyan?",
    ],
    "political_opinion": [
        "Politikai párt számlájára irányuló utalásnak van-e összeghatára?",
        "Kampánytámogatás utalni akarok — milyen közleményt adjak meg?",
    ],
    "trade_union": [
        "Szakszervezeti tagdíjat csoportos megbízással is fizethetek?",
        "Ha munkahelyet váltok, hogyan módosítom a szakszervezeti tagdíj utalás célszámláját?",
    ],
    "genetic": [
        "BRCA-pozitív teszt esetén a bank biztosítóján keresztül milyen fedezet érhető el?",
        "Örökletes betegség kockázatát genetikai igazolással el kell fogadni a hitelbírálatnál?",
    ],
}

# ── Multi-sentence context / story-like SPD ──
CONTEXTUAL_SPD: dict[str, list[str]] = {
    "health": [
        "Tavaly diagnosztizáltak szívbetegséggel. Azóta nem dolgozom teljes munkaidőben. Kérek törlesztési haladékot.",
        "A kezelésem hat hónapig tartott, a kórházi számlák felemésztették a megtakarításomat. Hitelt szeretnék.",
        "Három éve cukorbeteg vagyok, az inzulin drága. A jövedelemkiegészítő hitelt már meglévő igazolással igényelném.",
        "A műtét sikeresen zajlott, de rehabilitációm még tart. A bank felé is jelzem, hogy ez hat a fizetési képességemre.",
    ],
    "sex_life_orientation": [
        "Tíz éve élünk együtt a párommal. Melegek vagyunk. Most közös lakást vennénk és hitelre lenne szükségünk.",
        "A partnerem nevet változtatott és transznemű. A bankban még a régi neve szerepel, ezt szeretnénk rendezni.",
        "Bejegyzett élettársak vagyunk. A közös számla megnyitása körül sok a papírmunka, kérem segítsenek.",
    ],
    "religion_belief": [
        "Református vagyok, és minden hónapban adományozom a gyülekezetnek. A csoportos megbízás leállt.",
        "Új városba költöztünk. A helyi zsinagóga számlájára is szeretnék rendszeres adományt beállítani.",
        "Buddhista meditációs közösségnek adományozom az összeget. Kérem igazolják az utalás sikerét.",
    ],
    "ethnicity": [
        "Görög–magyar kettős állampolgár vagyok. Görögországi rokonaimnak rendszeres utalást szeretnék.",
        "A roma közösségünk fejlesztési alapjába havonta befizetek. A banki megbízást szeretném automatizálni.",
    ],
    "trade_union": [
        "Tíz éve szakszervezeti tag vagyok. Most állást változtattam, az új számlámról kell fizetni a tagdíjat.",
        "A Vasas szakszervezet tagjaként a tagdíjat és a sztrájkalapot is utalni kell. Hogyan teszem egy tranzakcióban?",
    ],
    "political_opinion": [
        "Az ellenzéki pártot évek óta támogatom. A kampányhoz most nagyobb összeget adományoznék.",
        "Zöldpárti vagyok. A párt helyi szervezetének számlájára havonta utalnék, de nem találom a kedvezményezettet.",
    ],
    "genetic": [
        "Az édesanyám BRCA-pozitív volt. Én is elvégeztem a tesztet, és pozitív lettem. Biztosítást kérek.",
        "Örökletes koleszterinbetegségemről 2022-ben kaptam diagnózist. Ez számít a hitelelbírálasnál?",
    ],
}

# ── Diverse extra negatives (no SPD, realistic banking frustrations) ──
DIVERSE_NEGATIVES: list[str] = [
    # Investment & savings
    "Az állampapír lejárt, hogyan újítom meg automatikusan?",
    "A befektetési alapom értéke csökkent, mikor érdemes kiszállni?",
    "Értékpapírszámla nyitáshoz milyen dokumentum kell?",
    "A részvényeladás után mennyi idő múlva kerül jóvá az összeg?",
    "Nem látom a kamatjóváírást a lekötött betétemben, miért?",
    "Az önkéntes nyugdíjpénztári számla összegét mikor vehetem fel?",
    "A megtakarítási tervem szerint 2027-re szeretnék X összeget összegyűjteni.",
    # Insurance linked to banking
    "A banki hitelfedezeti biztosítás mire terjed ki pontosan?",
    "Utazási biztosítás igénylése kártyás vásárlásnál automatikus?",
    "A hitelkártyám mellé jár-e vásárlóvédelmi biztosítás?",
    # Business edge cases
    "Katás vállalkozóként van-e kedvezményes számlacsomag?",
    "Az EVA-s vállalkozó devizaszámlát nyithat-e kedvezményesen?",
    "Könyvelőm időnként belép a nevemelben — hogyan adjak ideiglenes jogosultságot?",
    "A cégem megszűnt, a fennmaradó egyenleget magánszemélyként vehetem fel?",
    "Webáruházhoz integrált fizetési kapu bevezetéséhez kinek kell szólnom?",
    # Real estate / notary
    "Adásvételi szerződéshez letéti számlát kellene nyitni — hogyan?",
    "A közjegyző kért bankigazolást az ingatlan vásárláshoz.",
    "Az öröklési eljáráshoz a banknak milyen dokumentumot kell adni?",
    "Jelzálogjog bejegyzéséhez milyen banki nyilatkozat kell?",
    "Milyen díja van a banki letéti számla kezelésének?",
    # Daily digital
    "Az appban nem tudok névjegykártyát menteni a chathez.",
    "A bank weboldalán nem működik a jelszóemlékeztető link.",
    "Az e-mail értesítések nem érkeznek, pedig engedélyeztem őket.",
    "A netbanknál a szesszió 5 perc múlva lejár, hosszabbítható?",
    "Az Apple Pay-hez miért kér újra azonosítást minden reggel?",
    # Cross-border / expat
    "Külföldre költözök, meg kell szüntetni a magyar számlámat?",
    "Ausztráliából hogyan tudok forintban utalni haza?",
    "Kettős adózási egyezmény alapján milyen igazolást ad a bank?",
    "Az EU-s munkáltatóm euróban fizet, hogyan kapom forintban?",
    "Érvényes-e a magyar bankkártya Dél-Amerikában?",
    # Inheritance / estate
    "Elhunyt szülőm számlájához meghatalmazottként férhetek hozzá?",
    "A hagyatéki eljárás lezárása előtt ki kezeli a számlát?",
    "A néhai nagymamám betéte mi lesz, ha örökösök vagyunk?",
    "Az örökösödési adót a bankszámláról levonják automatikusan?",
    # Long-form negatives
    (
        "Kérem segítsenek! Három hónapja próbálom az örökségi pénzt átutalni a testvéremnek, "
        "de minden alkalommal 'limit túllépés' hibát kapok annak ellenére, hogy a napi utalási "
        "limitemet felemeltük. Telefonon azt mondták, tegyük számlán belüli átutalásnak, "
        "az sem működött. Mikor oldódik meg végre?"
    ),
    (
        "Üdvözlöm! Webáruházon keresztül vásároltam egy terméket, amely sohasem érkezett meg. "
        "A kereskedő visszafizetést ígért, de három hét telt el. Szeretnék visszaterhelést "
        "indítani a kártyán. Pontosan milyen dokumentumot kell feltöltenem, és mennyi ideig tart?"
    ),
    (
        "Autóhitelt vennék fel, de az online igénylő felület minden alkalommal leáll a "
        "jövedelemigazolás feltöltésénél. Fiókba mentem, de azt mondták, csak online lehet. "
        "Három hete nem jutok előre. Valaki tud segíteni, hogy eljussak a bírálatig?"
    ),
    (
        "Külföldi munkahelyről havonta kapok eurós utalást. Az összeg napokig feldolgozás "
        "alatt van, és más bankoknál ez nem volt így. Hogyan gyorsítható a jóváírás? "
        "Van-e SWIFT-nyomkövetési lehetőség az appban?"
    ),
    (
        "Az appban az összes korábbi tranzakciós kimutatásom eltűnt 2023 előttről. "
        "A könyvelőm az elmúlt három évről kér adatot az adóbevalláshoz. Hogyan tudom "
        "pótolni a hiányzó kimutatásokat?"
    ),
]

# ── Mixed-topic negatives (banking issue + SPD-adjacent, but label is negative) ──
# These test the model's ability NOT to trigger on coincidental mentions.
TRICKY_NEGATIVES: list[str] = [
    "A gyámolt unokám neve után nyitnék számlát, nem az enyéme az egészség.",  # 'egészség' in sentence but not SPD
    "Az apám is ide bankolt, most ő ajánlotta a bankot, semmi más indok.",
    "Beteg volt a macskám múlt héten, ezért nem tudtam időpontot foglalni a fiókba.",  # beteg = sick but not SPD
    "A templomunk közelében van az ATM, oda járok pénzt felvenni.",  # templomunk ≠ donation
    "A szakszervezeti meccsen nyertünk egy LCD tévét, hogyan adózom utána?",  # szakszervezet in neutral context
    "Az édesapám neve Smith János, ő küld euróban, de én fogadom forintban.",
    "A pártomat megkértük, hogy írja alá a közös bérleti szerződést.",  # 'párt' as partner, not political party
    "A munkáltatóm betegszabadság idején is fizeti a kártyajárulékot.",  # betegszabadság but no SPD detail
    "Genetika tantárgyból tutorálok középiskolásokat, fizetnek-e utánam járulékot?",  # genetika in unrelated context
    "A hitelem neve 'Személyi Egészség Plusz', ennek mi a kamat?",  # product name, not disclosure
    "A barátaim legtöbbje református, a közös bulira szeretnénk gyűjteni.",  # egyház-mention neutral
    "Melegebb lett az idő, de a fűtési számla mégis magas.",  # meleg = warm weather
    "Olvastam, hogy a roma munkalehetőségek nőnek, érintett ez a hitelprogramot?",  # news reference, not self-disclosure
    "A LGBT-szimbolika feltüntethető a kártyám képén?",  # support symbol, not orientation disclosure
    "A táppénzem összegéről kérek igazolást, mert albérlethez kell.",  # táppénz without health category detail
]
