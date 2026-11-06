"""Base fictive de plaintes consommateurs, de même forme que l'export réel.

Un tableau Excel (.xlsx), comme l'export réel : deux lignes vides, les en-têtes en ligne 3
(Case Number, Verbatim, Brand, Business Segment, Year Period, Blank), les données, quelques lignes
vides à la fin. Formats repris de l'export : verbatim préfixé par EXTERNAL, phrases souvent collées,
produits masqués par des étoiles, « Brand 001 », « Year Period » écrit « 26 P06 », Blank = « ■ ».
Aucune donnée réelle n'a servi : les textes sont composés à partir de phrases écrites à la main, en
anglais, français et allemand (les langues attestées dans l'export), et les marques sont de simples
numéros. D'autres langues sont disponibles par l'option --langues.

Des incidents sont cachés dans la base (même problème, même marque, même période, plusieurs langues),
ainsi qu'un pic saisonnier et une campagne de messages presque identiques, qui ne sont pas
des incidents. Leur liste est écrite à part, dans
verite/verite_incidents.csv, pour mesurer si une méthode les retrouve. Ce fichier ne fait pas partie
de la base et ne va pas aux étudiants.

    python generer_base_fictive.py                  # écrit dans ../donnees_fictives/
    python generer_base_fictive.py --total 6000 --graine 7

Bibliothèque standard, plus openpyxl pour écrire le tableau Excel.
"""
import argparse
import csv
import pathlib
import random

# ── Réglages de forme, à aligner sur la base réelle quand on connaîtra ses formats exacts ──────────
# « Year Period » s'écrit « 26 P06 » : année sur deux chiffres, période de P01 à P13
PERIODES = [(25, k) for k in range(1, 14)] + [(26, k) for k in range(1, 11)]
MASQUE_PRODUIT = "*******"             # les produits cités sont remplacés par des étoiles
PREFIXE = "EXTERNAL"                   # collé au début de chaque verbatim, comme dans l'export réel
CASE_MIN, CASE_MAX = 10_000_000, 10_199_999   # « Case Number » : huit chiffres
BLANK = "\u25a0"                       # la colonne Blank contient un seul caractère « ■ »

# segments et nombre de marques par segment (seul « Chocolate » est attesté ; les autres sont une hypothèse)
SEGMENTS = {"Chocolate": 6, "Confectionery": 4, "Petcare": 6, "Food": 4}
# langues attestées dans l'export : anglais, français, allemand ; les autres phrases restent disponibles
# (option --langues) au cas où la base réelle en contiendrait
LANGUES = {"en": 0.40, "fr": 0.32, "de": 0.28}
POIDS_AUTRES = {"es": 0.10, "it": 0.10, "nl": 0.07, "pl": 0.07, "ar": 0.08, "tr": 0.06}

# ── Phrases, par langue ─────────────────────────────────────────────────────────────────────────────
# {p} est remplacé par le produit, qui contient le masque d'anonymisation.
P = MASQUE_PRODUIT
TEXTES = {
 "fr": {
  "produits": {"Snacking": [f"une barre chocolatée {P}", f"un paquet de bonbons {P}", f"un sachet de chewing-gums {P}", f"une tablette {P}"],
               "Petcare": [f"un sac de croquettes {P}", f"des sachets de pâtée {P}", f"des friandises pour chien {P}"],
               "Food & Nutrition": [f"un paquet de riz {P}", f"un bocal de sauce {P}", f"un sachet de riz express {P}"]},
  "ouverture": ["J'ai acheté {p} la semaine dernière.", "Bonjour, j'ai acheté {p} au supermarché.", "Je vous écris au sujet de {p}.", "Hier j'ai ouvert {p}.", "Mon fils a mangé {p} ce matin."],
  "ouverture_animal": ["J'ai acheté {p} pour mon chat.", "Je donne {p} à mon chien depuis des années.", "Bonjour, j'ai acheté {p} pour mes deux chats."],
  "gout": ["Le goût était vraiment bizarre, presque amer.", "Le produit avait un goût de rance.", "Il n'avait pas du tout le goût habituel."],
  "odeur": ["En ouvrant le paquet il y avait une odeur désagréable.", "Le produit sentait le moisi.", "L'odeur était très forte et chimique."],
  "corps": ["J'ai trouvé un cheveu dans le produit.", "Il y avait un petit morceau de métal à l'intérieur.", "J'ai trouvé quelque chose de dur, comme un caillou."],
  "emballage": ["L'emballage était déchiré.", "Le paquet était ouvert en arrivant chez moi.", "La boîte était écrasée et le contenu abîmé."],
  "peremption": ["Le produit était moisi alors que la date n'était pas dépassée.", "Il y avait des taches blanches dessus.", "La date de péremption était déjà passée."],
  "texture": ["La texture était dure comme de la pierre.", "Le produit était complètement sec.", "C'était collant et mou."],
  "quantite": ["Il manquait plusieurs pièces dans le paquet.", "Le paquet était à moitié vide.", "Le poids ne correspond pas à ce qui est écrit."],
  "allergie": ["Ma fille a fait une réaction allergique après en avoir mangé.", "Il y avait des traces de noix qui ne sont pas indiquées.", "J'ai eu des démangeaisons juste après."],
  "animal": ["Mon chat refuse de manger depuis.", "Mon chien a eu la diarrhée deux jours.", "Mon chat a vomi après le repas."],
  "prix": ["La promotion n'a pas été appliquée en caisse.", "Le prix a beaucoup augmenté pour une plus petite quantité.", "Le bon de réduction n'a pas fonctionné."],
  "compliment": ["Je voulais simplement dire que c'est notre produit préféré.", "Merci, la nouvelle recette est excellente.", "Bravo pour la qualité, toute la famille adore."],
  "fondu": ["Le chocolat était complètement fondu à l'ouverture.", "Tout avait fondu et collait à l'emballage.", "Le produit était déformé par la chaleur."],
  "savon": ["Le chocolat avait un goût de savon.", "Ça sentait le savon, impossible à manger.", "Un goût de lessive, comme du savon, très désagréable."],
  "plastique": ["J'ai trouvé des petits morceaux de plastique bleu dans le riz.", "Il y avait des fils de plastique bleu dans le paquet.", "Un éclat de plastique bleu était mélangé au produit."],
  "recette": ["Depuis la nouvelle recette mon chat vomit après chaque repas.", "Avec le nouveau paquet mon chat refuse de manger et vomit.", "La nouvelle formule rend mon chat malade."],
  "emotion": ["Je suis très déçue.", "C'est inadmissible.", "Je suis vraiment inquiète.", "J'espère que vous allez faire quelque chose.", ""],
  "demande": ["Je voudrais être remboursée.", "Merci de me contacter.", "Pouvez-vous m'expliquer ce qui s'est passé ?", "Je n'achèterai plus ce produit.", ""],
 },
 "en": {
  "produits": {"Snacking": [f"a chocolate bar {P}", f"a bag of sweets {P}", f"a pack of chewing gum {P}", f"a {P} chocolate bar"],
               "Petcare": [f"a bag of dry cat food {P}", f"some pouches of wet food {P}", f"dog treats {P}"],
               "Food & Nutrition": [f"a bag of rice {P}", f"a jar of {P} sauce", f"a pouch of microwave rice {P}"]},
  "ouverture": ["I bought {p} last week.", "Hi, I purchased {p} at my local store.", "I am writing about {p}.", "Yesterday I opened {p}.", "My son ate {p} this morning."],
  "ouverture_animal": ["I bought {p} for my cat.", "I have been feeding {p} to my dog for years.", "Hello, I bought {p} for my two cats."],
  "gout": ["The taste was really strange, almost bitter.", "It tasted rancid.", "It did not taste like it usually does."],
  "odeur": ["When I opened the pack there was an unpleasant smell.", "The product smelled mouldy.", "The smell was very strong and chemical."],
  "corps": ["I found a hair in the product.", "There was a small piece of metal inside.", "I found something hard, like a stone."],
  "emballage": ["The packaging was torn.", "The pack was already open when I got home.", "The box was crushed and the contents damaged."],
  "peremption": ["The product was mouldy even though it was before the best before date.", "There were white spots on it.", "It was already past its expiry date."],
  "texture": ["The texture was rock hard.", "The product was completely dry.", "It was sticky and soft."],
  "quantite": ["Several pieces were missing from the pack.", "The bag was half empty.", "The weight does not match the label."],
  "allergie": ["My daughter had an allergic reaction after eating it.", "There were traces of nuts that are not on the label.", "I got itchy right after eating it."],
  "animal": ["My cat has refused to eat since.", "My dog had diarrhoea for two days.", "My cat threw up after the meal."],
  "prix": ["The promotion was not applied at the till.", "The price went up a lot for a smaller pack.", "The coupon did not work."],
  "compliment": ["I just wanted to say this is our favourite product.", "Thank you, the new recipe is excellent.", "Great quality, the whole family loves it."],
  "fondu": ["The chocolate was completely melted when I opened it.", "Everything had melted and stuck to the wrapper.", "The product was misshapen from the heat."],
  "savon": ["The chocolate tasted of soap.", "It smelled like soap, impossible to eat.", "A soapy taste, like washing powder, really unpleasant."],
  "plastique": ["I found small pieces of blue plastic in the rice.", "There were blue plastic threads in the pack.", "A shard of blue plastic was mixed in with the product."],
  "recette": ["Since the new recipe my cat vomits after every meal.", "With the new pack my cat refuses to eat and throws up.", "The new formula is making my cat ill."],
  "emotion": ["I am very disappointed.", "This is unacceptable.", "I am really worried.", "I hope you will do something about it.", ""],
  "demande": ["I would like a refund.", "Please contact me.", "Can you explain what happened?", "I will not buy this product again.", ""],
 },
 "de": {
  "produits": {"Snacking": [f"einen Schokoriegel {P}", f"eine Tüte Bonbons {P}", f"eine Packung Kaugummi {P}", f"eine Tafel {P}"],
               "Petcare": [f"einen Sack Trockenfutter {P}", f"Nassfutterbeutel {P}", f"Hundesnacks {P}"],
               "Food & Nutrition": [f"eine Packung Reis {P}", f"ein Glas Soße {P}", f"einen Beutel Express-Reis {P}"]},
  "ouverture": ["Ich habe letzte Woche {p} gekauft.", "Guten Tag, ich habe {p} im Supermarkt gekauft.", "Ich schreibe Ihnen wegen {p}.", "Gestern habe ich {p} geöffnet.", "Mein Sohn hat heute Morgen {p} gegessen."],
  "ouverture_animal": ["Ich habe {p} für meine Katze gekauft.", "Ich füttere meinen Hund seit Jahren mit {p}.", "Hallo, ich habe {p} für meine zwei Katzen gekauft."],
  "gout": ["Der Geschmack war sehr seltsam, fast bitter.", "Es schmeckte ranzig.", "Es schmeckte überhaupt nicht wie sonst."],
  "odeur": ["Beim Öffnen gab es einen unangenehmen Geruch.", "Das Produkt roch muffig.", "Der Geruch war sehr stark und chemisch."],
  "corps": ["Ich habe ein Haar im Produkt gefunden.", "Es war ein kleines Metallstück darin.", "Ich habe etwas Hartes gefunden, wie ein Stein."],
  "emballage": ["Die Verpackung war zerrissen.", "Die Packung war schon offen, als ich nach Hause kam.", "Die Schachtel war zerdrückt und der Inhalt beschädigt."],
  "peremption": ["Das Produkt war schimmelig, obwohl das Datum nicht abgelaufen war.", "Es waren weiße Flecken darauf.", "Das Mindesthaltbarkeitsdatum war schon abgelaufen."],
  "texture": ["Die Konsistenz war steinhart.", "Das Produkt war völlig trocken.", "Es war klebrig und weich."],
  "quantite": ["Es fehlten mehrere Stücke in der Packung.", "Die Tüte war halb leer.", "Das Gewicht stimmt nicht mit der Angabe überein."],
  "allergie": ["Meine Tochter hatte nach dem Essen eine allergische Reaktion.", "Es waren Spuren von Nüssen darin, die nicht angegeben sind.", "Ich hatte direkt danach Juckreiz."],
  "animal": ["Meine Katze frisst seitdem nicht mehr.", "Mein Hund hatte zwei Tage Durchfall.", "Meine Katze hat sich nach dem Fressen übergeben."],
  "prix": ["Das Angebot wurde an der Kasse nicht berücksichtigt.", "Der Preis ist für eine kleinere Packung stark gestiegen.", "Der Gutschein hat nicht funktioniert."],
  "compliment": ["Ich wollte nur sagen, dass das unser Lieblingsprodukt ist.", "Danke, das neue Rezept ist ausgezeichnet.", "Tolle Qualität, die ganze Familie liebt es."],
  "fondu": ["Die Schokolade war beim Öffnen komplett geschmolzen.", "Alles war geschmolzen und klebte an der Verpackung.", "Das Produkt war durch die Hitze verformt."],
  "savon": ["Die Schokolade schmeckte nach Seife.", "Es roch nach Seife, nicht essbar.", "Ein seifiger Geschmack, wie Waschmittel, sehr unangenehm."],
  "plastique": ["Ich habe kleine blaue Plastikstücke im Reis gefunden.", "In der Packung waren blaue Plastikfäden.", "Ein Splitter aus blauem Plastik war im Produkt."],
  "recette": ["Seit dem neuen Rezept erbricht meine Katze nach jeder Mahlzeit.", "Mit der neuen Packung frisst meine Katze nicht und erbricht.", "Die neue Rezeptur macht meine Katze krank."],
  "emotion": ["Ich bin sehr enttäuscht.", "Das ist inakzeptabel.", "Ich mache mir wirklich Sorgen.", "Ich hoffe, Sie unternehmen etwas.", ""],
  "demande": ["Ich möchte mein Geld zurück.", "Bitte kontaktieren Sie mich.", "Können Sie mir erklären, was passiert ist?", "Ich werde dieses Produkt nicht mehr kaufen.", ""],
 },
 "es": {
  "produits": {"Snacking": [f"una barrita de chocolate {P}", f"una bolsa de caramelos {P}", f"un paquete de chicles {P}", f"una tableta {P}"],
               "Petcare": [f"un saco de pienso para gatos {P}", f"unas bolsitas de comida húmeda {P}", f"premios para perro {P}"],
               "Food & Nutrition": [f"un paquete de arroz {P}", f"un bote de salsa {P}", f"un vasito de arroz {P}"]},
  "ouverture": ["Compré {p} la semana pasada.", "Hola, compré {p} en el supermercado.", "Les escribo por {p}.", "Ayer abrí {p}.", "Mi hijo comió {p} esta mañana."],
  "ouverture_animal": ["Compré {p} para mi gato.", "Le doy {p} a mi perro desde hace años.", "Hola, compré {p} para mis dos gatos."],
  "gout": ["El sabor era muy raro, casi amargo.", "Sabía a rancio.", "No tenía el sabor de siempre."],
  "odeur": ["Al abrir el paquete había un olor desagradable.", "El producto olía a humedad.", "El olor era muy fuerte y químico."],
  "corps": ["Encontré un pelo en el producto.", "Había un trozo pequeño de metal dentro.", "Encontré algo duro, como una piedra."],
  "emballage": ["El envase estaba roto.", "El paquete ya estaba abierto al llegar a casa.", "La caja estaba aplastada y el contenido dañado."],
  "peremption": ["El producto tenía moho aunque no estaba caducado.", "Tenía manchas blancas.", "Ya había pasado la fecha de caducidad."],
  "texture": ["La textura era dura como una piedra.", "El producto estaba completamente seco.", "Estaba pegajoso y blando."],
  "quantite": ["Faltaban varias piezas en el paquete.", "La bolsa estaba medio vacía.", "El peso no corresponde con la etiqueta."],
  "allergie": ["Mi hija tuvo una reacción alérgica después de comerlo.", "Había trazas de frutos secos que no están indicadas.", "Me picaba todo justo después."],
  "animal": ["Mi gato no quiere comer desde entonces.", "Mi perro tuvo diarrea dos días.", "Mi gato vomitó después de comer."],
  "prix": ["La promoción no se aplicó en caja.", "El precio subió mucho para un paquete más pequeño.", "El cupón no funcionó."],
  "compliment": ["Solo quería decir que es nuestro producto favorito.", "Gracias, la nueva receta está buenísima.", "Muy buena calidad, a toda la familia le encanta."],
  "fondu": ["El chocolate estaba totalmente derretido al abrirlo.", "Todo se había derretido y estaba pegado al envoltorio.", "El producto estaba deformado por el calor."],
  "savon": ["El chocolate sabía a jabón.", "Olía a jabón, imposible de comer.", "Un sabor a jabón, como a detergente, muy desagradable."],
  "plastique": ["Encontré trocitos de plástico azul en el arroz.", "Había hilos de plástico azul en el paquete.", "Un fragmento de plástico azul estaba mezclado con el producto."],
  "recette": ["Desde la nueva receta mi gato vomita después de cada comida.", "Con el nuevo paquete mi gato no come y vomita.", "La nueva fórmula pone enfermo a mi gato."],
  "emotion": ["Estoy muy decepcionada.", "Es inaceptable.", "Estoy realmente preocupada.", "Espero que hagan algo.", ""],
  "demande": ["Quiero que me devuelvan el dinero.", "Por favor, contáctenme.", "¿Pueden explicarme qué ha pasado?", "No volveré a comprar este producto.", ""],
 },
 "it": {
  "produits": {"Snacking": [f"una barretta di cioccolato {P}", f"un sacchetto di caramelle {P}", f"un pacchetto di gomme {P}", f"una tavoletta {P}"],
               "Petcare": [f"un sacco di crocchette {P}", f"delle bustine di umido {P}", f"snack per cani {P}"],
               "Food & Nutrition": [f"un pacco di riso {P}", f"un barattolo di sugo {P}", f"una busta di riso pronto {P}"]},
  "ouverture": ["Ho comprato {p} la settimana scorsa.", "Buongiorno, ho comprato {p} al supermercato.", "Vi scrivo a proposito di {p}.", "Ieri ho aperto {p}.", "Mio figlio ha mangiato {p} stamattina."],
  "ouverture_animal": ["Ho comprato {p} per il mio gatto.", "Do {p} al mio cane da anni.", "Salve, ho comprato {p} per i miei due gatti."],
  "gout": ["Il sapore era molto strano, quasi amaro.", "Sapeva di rancido.", "Non aveva il solito sapore."],
  "odeur": ["Aprendo la confezione c'era un odore sgradevole.", "Il prodotto sapeva di muffa.", "L'odore era fortissimo e chimico."],
  "corps": ["Ho trovato un capello nel prodotto.", "C'era un pezzetto di metallo dentro.", "Ho trovato qualcosa di duro, come un sassolino."],
  "emballage": ["La confezione era strappata.", "Il pacco era già aperto quando sono arrivata a casa.", "La scatola era schiacciata e il contenuto rovinato."],
  "peremption": ["Il prodotto era ammuffito anche se non era scaduto.", "C'erano delle macchie bianche.", "La data di scadenza era già passata."],
  "texture": ["La consistenza era dura come un sasso.", "Il prodotto era completamente secco.", "Era appiccicoso e molle."],
  "quantite": ["Mancavano diversi pezzi nella confezione.", "Il sacchetto era mezzo vuoto.", "Il peso non corrisponde a quello indicato."],
  "allergie": ["Mia figlia ha avuto una reazione allergica dopo averlo mangiato.", "C'erano tracce di frutta a guscio non indicate.", "Ho avuto prurito subito dopo."],
  "animal": ["Il mio gatto da allora non mangia più.", "Il mio cane ha avuto la diarrea per due giorni.", "Il mio gatto ha vomitato dopo il pasto."],
  "prix": ["La promozione non è stata applicata alla cassa.", "Il prezzo è aumentato molto per una confezione più piccola.", "Il buono sconto non ha funzionato."],
  "compliment": ["Volevo solo dire che è il nostro prodotto preferito.", "Grazie, la nuova ricetta è ottima.", "Qualità eccellente, piace a tutta la famiglia."],
  "fondu": ["Il cioccolato era completamente sciolto all'apertura.", "Si era sciolto tutto ed era attaccato all'involucro.", "Il prodotto era deformato dal caldo."],
  "savon": ["Il cioccolato sapeva di sapone.", "Odorava di sapone, impossibile da mangiare.", "Un sapore di sapone, come di detersivo, molto sgradevole."],
  "plastique": ["Ho trovato dei pezzetti di plastica blu nel riso.", "C'erano dei fili di plastica blu nella confezione.", "Una scheggia di plastica blu era mescolata al prodotto."],
  "recette": ["Dalla nuova ricetta il mio gatto vomita dopo ogni pasto.", "Con la nuova confezione il mio gatto non mangia e vomita.", "La nuova formula fa stare male il mio gatto."],
  "emotion": ["Sono molto delusa.", "È inaccettabile.", "Sono davvero preoccupata.", "Spero che facciate qualcosa.", ""],
  "demande": ["Vorrei essere rimborsata.", "Vi prego di contattarmi.", "Potete spiegarmi cosa è successo?", "Non comprerò più questo prodotto.", ""],
 },
 "nl": {
  "produits": {"Snacking": [f"een chocoladereep {P}", f"een zak snoep {P}", f"een pakje kauwgom {P}", f"een reep {P}"],
               "Petcare": [f"een zak droogvoer {P}", f"zakjes natvoer {P}", f"hondensnacks {P}"],
               "Food & Nutrition": [f"een pak rijst {P}", f"een pot saus {P}", f"een zakje magnetronrijst {P}"]},
  "ouverture": ["Ik heb vorige week {p} gekocht.", "Hallo, ik heb {p} in de supermarkt gekocht.", "Ik schrijf u over {p}.", "Gisteren heb ik {p} geopend.", "Mijn zoon heeft vanochtend {p} gegeten."],
  "ouverture_animal": ["Ik heb {p} voor mijn kat gekocht.", "Ik geef mijn hond al jaren {p}.", "Hallo, ik heb {p} voor mijn twee katten gekocht."],
  "gout": ["De smaak was heel vreemd, bijna bitter.", "Het smaakte ranzig.", "Het smaakte helemaal niet zoals normaal."],
  "odeur": ["Bij het openen was er een vieze geur.", "Het product rook muf.", "De geur was erg sterk en chemisch."],
  "corps": ["Ik vond een haar in het product.", "Er zat een klein stukje metaal in.", "Ik vond iets hards, zoals een steentje."],
  "emballage": ["De verpakking was gescheurd.", "Het pak was al open toen ik thuiskwam.", "De doos was geplet en de inhoud beschadigd."],
  "peremption": ["Het product was beschimmeld terwijl de datum nog niet verstreken was.", "Er zaten witte vlekken op.", "De houdbaarheidsdatum was al verstreken."],
  "texture": ["De structuur was keihard.", "Het product was helemaal droog.", "Het was plakkerig en zacht."],
  "quantite": ["Er ontbraken meerdere stukken in het pak.", "De zak was half leeg.", "Het gewicht klopt niet met het etiket."],
  "allergie": ["Mijn dochter kreeg een allergische reactie na het eten.", "Er zaten sporen van noten in die niet vermeld staan.", "Ik kreeg meteen jeuk."],
  "animal": ["Mijn kat wil sindsdien niet meer eten.", "Mijn hond had twee dagen diarree.", "Mijn kat moest overgeven na het eten."],
  "prix": ["De aanbieding werd niet verrekend aan de kassa.", "De prijs is flink gestegen voor een kleiner pak.", "De kortingsbon werkte niet."],
  "compliment": ["Ik wilde alleen zeggen dat dit ons favoriete product is.", "Bedankt, het nieuwe recept is heerlijk.", "Geweldige kwaliteit, het hele gezin is er dol op."],
  "fondu": ["De chocolade was helemaal gesmolten bij het openen.", "Alles was gesmolten en plakte aan de wikkel.", "Het product was vervormd door de hitte."],
  "savon": ["De chocolade smaakte naar zeep.", "Het rook naar zeep, niet te eten.", "Een zeepsmaak, als wasmiddel, heel onaangenaam."],
  "plastique": ["Ik vond kleine stukjes blauw plastic in de rijst.", "Er zaten blauwe plastic draadjes in het pak.", "Een splinter blauw plastic zat tussen het product."],
  "recette": ["Sinds het nieuwe recept braakt mijn kat na elke maaltijd.", "Met de nieuwe verpakking eet mijn kat niet en braakt ze.", "De nieuwe formule maakt mijn kat ziek."],
  "emotion": ["Ik ben erg teleurgesteld.", "Dit is onaanvaardbaar.", "Ik maak me echt zorgen.", "Ik hoop dat u er iets aan doet.", ""],
  "demande": ["Ik wil graag mijn geld terug.", "Neem alstublieft contact met mij op.", "Kunt u uitleggen wat er gebeurd is?", "Ik koop dit product niet meer.", ""],
 },
 "pl": {
  "produits": {"Snacking": [f"baton czekoladowy {P}", f"paczkę cukierków {P}", f"opakowanie gumy do żucia {P}", f"tabliczkę {P}"],
               "Petcare": [f"worek suchej karmy {P}", f"saszetki mokrej karmy {P}", f"przysmaki dla psa {P}"],
               "Food & Nutrition": [f"paczkę ryżu {P}", f"słoik sosu {P}", f"saszetkę ryżu {P}"]},
  "ouverture": ["W zeszłym tygodniu kupiłam {p}.", "Dzień dobry, kupiłam {p} w supermarkecie.", "Piszę w sprawie produktu {p}.", "Wczoraj otworzyłam {p}.", "Mój syn zjadł dziś rano {p}."],
  "ouverture_animal": ["Kupiłam {p} dla mojego kota.", "Od lat karmię psa produktem {p}.", "Dzień dobry, kupiłam {p} dla moich dwóch kotów."],
  "gout": ["Smak był bardzo dziwny, prawie gorzki.", "Produkt był zjełczały.", "Smakował zupełnie inaczej niż zwykle."],
  "odeur": ["Po otwarciu był nieprzyjemny zapach.", "Produkt pachniał stęchlizną.", "Zapach był bardzo mocny i chemiczny."],
  "corps": ["Znalazłam włos w produkcie.", "W środku był mały kawałek metalu.", "Znalazłam coś twardego, jak kamyk."],
  "emballage": ["Opakowanie było rozdarte.", "Paczka była już otwarta, gdy wróciłam do domu.", "Pudełko było zgniecione, a zawartość uszkodzona."],
  "peremption": ["Produkt był spleśniały, chociaż data nie minęła.", "Były na nim białe plamy.", "Data ważności już minęła."],
  "texture": ["Konsystencja była twarda jak kamień.", "Produkt był zupełnie suchy.", "Był lepki i miękki."],
  "quantite": ["W opakowaniu brakowało kilku sztuk.", "Torebka była w połowie pusta.", "Waga nie zgadza się z etykietą."],
  "allergie": ["Moja córka miała reakcję alergiczną po zjedzeniu.", "Były ślady orzechów, których nie ma na etykiecie.", "Zaraz potem wszystko mnie swędziało."],
  "animal": ["Mój kot od tego czasu nie chce jeść.", "Mój pies miał biegunkę przez dwa dni.", "Mój kot zwymiotował po jedzeniu."],
  "prix": ["Promocja nie została naliczona przy kasie.", "Cena bardzo wzrosła za mniejsze opakowanie.", "Kupon nie zadziałał."],
  "compliment": ["Chciałam tylko powiedzieć, że to nasz ulubiony produkt.", "Dziękuję, nowy przepis jest wspaniały.", "Świetna jakość, cała rodzina to uwielbia."],
  "fondu": ["Czekolada była całkowicie roztopiona po otwarciu.", "Wszystko się roztopiło i przykleiło do opakowania.", "Produkt był zdeformowany od ciepła."],
  "savon": ["Czekolada smakowała jak mydło.", "Pachniało mydłem, nie da się tego jeść.", "Mydlany smak, jak proszek do prania, bardzo nieprzyjemny."],
  "plastique": ["Znalazłam małe kawałki niebieskiego plastiku w ryżu.", "W paczce były niebieskie plastikowe nitki.", "Odłamek niebieskiego plastiku był w produkcie."],
  "recette": ["Od nowego przepisu mój kot wymiotuje po każdym posiłku.", "Z nowym opakowaniem mój kot nie je i wymiotuje.", "Nowa receptura sprawia, że mój kot choruje."],
  "emotion": ["Jestem bardzo rozczarowana.", "To niedopuszczalne.", "Naprawdę się martwię.", "Mam nadzieję, że coś z tym zrobicie.", ""],
  "demande": ["Proszę o zwrot pieniędzy.", "Proszę o kontakt.", "Czy możecie wyjaśnić, co się stało?", "Nie kupię więcej tego produktu.", ""],
 },
 "ar": {
  "produits": {"Snacking": [f"لوح شوكولاتة {P}", f"كيس حلوى {P}", f"علبة علكة {P}", f"قطعة شوكولاتة {P}"],
               "Petcare": [f"كيس طعام جاف للقطط {P}", f"أكياس طعام رطب {P}", f"مكافآت للكلاب {P}"],
               "Food & Nutrition": [f"كيس أرز {P}", f"برطمان صلصة {P}", f"كيس أرز سريع التحضير {P}"]},
  "ouverture": ["اشتريت {p} الأسبوع الماضي.", "مرحبا، اشتريت {p} من السوبرماركت.", "أكتب إليكم بخصوص {p}.", "فتحت {p} أمس.", "أكل ابني {p} هذا الصباح."],
  "ouverture_animal": ["اشتريت {p} لقطتي.", "أطعم كلبي {p} منذ سنوات.", "مرحبا، اشتريت {p} لقطتيّ."],
  "gout": ["كان الطعم غريبا جدا، مرّا تقريبا.", "كان طعمه زنخا.", "لم يكن طعمه كالمعتاد."],
  "odeur": ["عند فتح العبوة كانت هناك رائحة كريهة.", "كانت رائحة المنتج عفنة.", "كانت الرائحة قوية جدا وكيميائية."],
  "corps": ["وجدت شعرة في المنتج.", "كانت هناك قطعة معدن صغيرة في الداخل.", "وجدت شيئا صلبا مثل الحجر."],
  "emballage": ["كانت العبوة ممزقة.", "كانت العلبة مفتوحة عندما وصلت إلى البيت.", "كان الصندوق مسحوقا والمحتوى تالفا."],
  "peremption": ["كان المنتج متعفنا رغم أن التاريخ لم ينته.", "كانت عليه بقع بيضاء.", "كان تاريخ الصلاحية قد انتهى."],
  "texture": ["كان القوام صلبا كالحجر.", "كان المنتج جافا تماما.", "كان لزجا وطريا."],
  "quantite": ["كانت هناك قطع ناقصة في العبوة.", "كان الكيس نصف فارغ.", "الوزن لا يطابق ما هو مكتوب."],
  "allergie": ["أصيبت ابنتي بحساسية بعد أن أكلته.", "كانت هناك آثار مكسرات غير مذكورة.", "شعرت بحكة مباشرة بعد ذلك."],
  "animal": ["قطتي ترفض الأكل منذ ذلك الوقت.", "أصيب كلبي بالإسهال يومين.", "تقيأت قطتي بعد الوجبة."],
  "prix": ["لم يتم تطبيق العرض عند الدفع.", "ارتفع السعر كثيرا لعبوة أصغر.", "لم تعمل القسيمة."],
  "compliment": ["أردت فقط أن أقول إنه منتجنا المفضل.", "شكرا، الوصفة الجديدة ممتازة.", "جودة رائعة، العائلة كلها تحبه."],
  "fondu": ["كانت الشوكولاتة ذائبة تماما عند الفتح.", "ذاب كل شيء والتصق بالغلاف.", "كان المنتج مشوها بسبب الحرارة."],
  "savon": ["كان طعم الشوكولاتة مثل الصابون.", "كانت رائحته مثل الصابون، لا يمكن أكله.", "طعم صابون مثل مسحوق الغسيل، مزعج جدا."],
  "plastique": ["وجدت قطعا صغيرة من البلاستيك الأزرق في الأرز.", "كانت هناك خيوط بلاستيك زرقاء في الكيس.", "كانت شظية من البلاستيك الأزرق مختلطة بالمنتج."],
  "recette": ["منذ الوصفة الجديدة تتقيأ قطتي بعد كل وجبة.", "مع العبوة الجديدة ترفض قطتي الأكل وتتقيأ.", "التركيبة الجديدة تجعل قطتي مريضة."],
  "emotion": ["أنا محبطة جدا.", "هذا غير مقبول.", "أنا قلقة حقا.", "أتمنى أن تفعلوا شيئا.", ""],
  "demande": ["أريد استرداد المبلغ.", "أرجو التواصل معي.", "هل يمكنكم أن تشرحوا ما حدث؟", "لن أشتري هذا المنتج مرة أخرى.", ""],
 },
 "tr": {
  "produits": {"Snacking": [f"bir çikolata {P}", f"bir paket şeker {P}", f"bir paket sakız {P}", f"bir tablet çikolata {P}"],
               "Petcare": [f"bir torba kuru kedi maması {P}", f"yaş mama poşetleri {P}", f"köpek ödülleri {P}"],
               "Food & Nutrition": [f"bir paket pirinç {P}", f"bir kavanoz sos {P}", f"bir poşet hazır pirinç {P}"]},
  "ouverture": ["Geçen hafta {p} aldım.", "Merhaba, marketten {p} aldım.", "{p} hakkında yazıyorum.", "Dün {p} açtım.", "Oğlum bu sabah {p} yedi."],
  "ouverture_animal": ["Kedim için {p} aldım.", "Köpeğime yıllardır {p} veriyorum.", "Merhaba, iki kedim için {p} aldım."],
  "gout": ["Tadı çok tuhaftı, neredeyse acıydı.", "Tadı bayattı.", "Her zamanki tadı yoktu."],
  "odeur": ["Paketi açınca kötü bir koku vardı.", "Ürün küf kokuyordu.", "Koku çok güçlü ve kimyasaldı."],
  "corps": ["Ürünün içinde bir saç buldum.", "İçinde küçük bir metal parçası vardı.", "Taş gibi sert bir şey buldum."],
  "emballage": ["Ambalaj yırtıktı.", "Eve geldiğimde paket zaten açıktı.", "Kutu ezilmişti ve içindekiler zarar görmüştü."],
  "peremption": ["Tarih geçmediği halde ürün küflüydü.", "Üzerinde beyaz lekeler vardı.", "Son kullanma tarihi geçmişti."],
  "texture": ["Kıvamı taş gibi sertti.", "Ürün tamamen kuruydu.", "Yapışkan ve yumuşaktı."],
  "quantite": ["Pakette birkaç parça eksikti.", "Torba yarı boştu.", "Ağırlık etiketteki ile uyuşmuyor."],
  "allergie": ["Kızım yedikten sonra alerjik reaksiyon gösterdi.", "Etikette yazmayan fındık izleri vardı.", "Hemen ardından kaşıntım oldu."],
  "animal": ["Kedim o zamandan beri yemek yemiyor.", "Köpeğim iki gün ishal oldu.", "Kedim yemekten sonra kustu."],
  "prix": ["Kampanya kasada uygulanmadı.", "Daha küçük paket için fiyat çok arttı.", "İndirim kuponu çalışmadı."],
  "compliment": ["Sadece bunun en sevdiğimiz ürün olduğunu söylemek istedim.", "Teşekkürler, yeni tarif harika.", "Kalite çok iyi, bütün aile bayılıyor."],
  "fondu": ["Açtığımda çikolata tamamen erimişti.", "Her şey erimiş ve ambalaja yapışmıştı.", "Ürün sıcaktan şekli bozulmuştu."],
  "savon": ["Çikolatanın tadı sabun gibiydi.", "Sabun kokuyordu, yenecek gibi değildi.", "Deterjan gibi sabunlu bir tat, çok rahatsız edici."],
  "plastique": ["Pirincin içinde küçük mavi plastik parçaları buldum.", "Pakette mavi plastik iplikler vardı.", "Ürünün içine mavi bir plastik kıymığı karışmıştı."],
  "recette": ["Yeni tariften beri kedim her yemekten sonra kusuyor.", "Yeni paketle kedim yemiyor ve kusuyor.", "Yeni formül kedimi hasta ediyor."],
  "emotion": ["Çok hayal kırıklığına uğradım.", "Bu kabul edilemez.", "Gerçekten endişeliyim.", "Umarım bir şey yaparsınız.", ""],
  "demande": ["Paramı geri istiyorum.", "Lütfen benimle iletişime geçin.", "Ne olduğunu açıklayabilir misiniz?", "Bu ürünü bir daha almayacağım.", ""],
 },
}

# Les produits ont été écrits par grand rayon ; on les range selon les segments de l'export.
for _t in TEXTES.values():
    _pr = _t["produits"]
    _s = _pr.pop("Snacking")
    _pr["Chocolate"], _pr["Confectionery"] = [_s[0], _s[3]], [_s[1], _s[2]]
    _pr["Food"] = _pr.pop("Food & Nutrition")

# formules de politesse : une lettre commence souvent par l'une et finit par l'autre
POLITESSE = {
    "fr": (["Bonjour,", "Madame, Monsieur,", "Bonjour à vous,"], ["Cordialement.", "Bien à vous.", "Merci d'avance."]),
    "en": (["Hello,", "Dear Sir or Madam,", "Hi,"], ["Kind regards.", "Best regards.", "Thanks."]),
    "de": (["Guten Tag,", "Sehr geehrte Damen und Herren,", "Hallo,"], ["Mit freundlichen Grüßen.", "Viele Grüße.", "Danke."]),
    "es": (["Hola,", "Buenos días,", "Estimados señores,"], ["Un saludo.", "Atentamente.", "Gracias."]),
    "it": (["Buongiorno,", "Gentili signori,", "Salve,"], ["Cordiali saluti.", "Grazie.", "Distinti saluti."]),
    "nl": (["Goedendag,", "Beste,", "Hallo,"], ["Met vriendelijke groet.", "Groeten.", "Bedankt."]),
    "pl": (["Dzień dobry,", "Szanowni Państwo,"], ["Pozdrawiam.", "Z poważaniem."]),
    "ar": (["مرحبا،", "السادة المحترمون،"], ["مع التحية.", "شكرا."]),
    "tr": (["Merhaba,", "Sayın yetkili,"], ["Saygılarımla.", "İyi günler."]),
}

# message de campagne : beaucoup de consommateurs envoient presque le même texte après avoir scanné un
# produit dans une application de notation ; rédaction et additif inventés pour la base fictive
CAMPAGNE = {
    "fr": "J'ai scanné le produit {m} avec une application de notation alimentaire. Elle signale un additif jugé à risque, le dioxyde de titane (E171).Cet additif est soupçonné d'effets nocifs sur la santé, en particulier chez les enfants, et il est déjà interdit dans certains pays.Voici la fiche du produit dans l'application : {u}Je vous demande de retirer cet additif de votre recette pour protéger les consommateurs.",
    "en": "I scanned the product {m} with a food rating app. It flags an additive rated as risky, titanium dioxide (E171).This additive is suspected of harmful health effects, especially for children, and is already banned in some countries.Here is the product page in the app: {u}I am asking you to remove this additive from your recipe to protect consumers.",
    "de": "Ich habe das Produkt {m} mit einer Lebensmittel-Bewertungs-App gescannt. Sie meldet einen als riskant eingestuften Zusatzstoff, Titandioxid (E171).Dieser Zusatzstoff steht im Verdacht, gesundheitsschädlich zu sein, besonders für Kinder, und ist in einigen Ländern bereits verboten.Hier die Produktseite in der App: {u}Ich bitte Sie, diesen Zusatzstoff aus Ihrem Rezept zu entfernen, um die Verbraucher zu schützen.",
    "es": "He escaneado el producto {m} con una aplicación de valoración de alimentos. Señala un aditivo considerado de riesgo, el dióxido de titanio (E171).Se sospecha que este aditivo es perjudicial para la salud, sobre todo para los niños, y ya está prohibido en algunos países.Esta es la ficha del producto en la aplicación: {u}Les pido que eliminen este aditivo de su receta para proteger a los consumidores.",
    "it": "Ho scansionato il prodotto {m} con un'app di valutazione degli alimenti. Segnala un additivo considerato a rischio, il biossido di titanio (E171).Questo additivo è sospettato di effetti nocivi sulla salute, soprattutto per i bambini, ed è già vietato in alcuni paesi.Ecco la scheda del prodotto nell'app: {u}Vi chiedo di eliminare questo additivo dalla vostra ricetta per proteggere i consumatori.",
}

# problèmes ordinaires, avec leur poids, selon le segment
_SUCRE = {"gout": 3, "odeur": 2, "corps": 2, "emballage": 3, "peremption": 2, "texture": 3, "quantite": 3,
          "allergie": 1, "prix": 2, "compliment": 1}
FOND = {
    "Chocolate": _SUCRE,
    "Confectionery": _SUCRE,
    "Petcare": {"odeur": 2, "corps": 2, "emballage": 3, "peremption": 2, "texture": 1, "quantite": 2, "animal": 4,
                "prix": 2, "compliment": 1},
    "Food": {"gout": 3, "odeur": 2, "corps": 2, "emballage": 3, "peremption": 3, "texture": 2, "quantite": 2,
             "allergie": 1, "prix": 2, "compliment": 1},
}

# groupes cachés ; n = nombre de plaintes pour une base de 1 301 lignes (la taille de l'échantillon réel)
INCIDENTS = [
    {"id": "I1_savon", "probleme": "savon", "segment": "Chocolate", "marque_rang": 3,
     "periodes": [(26, 5)], "langues": ["fr", "de", "en"], "n": 28},
    {"id": "I2_plastique", "probleme": "plastique", "segment": "Food", "marque_rang": 2,
     "periodes": [(25, 10)], "langues": ["fr", "de", "en"], "n": 20},
    {"id": "I3_recette", "probleme": "recette", "segment": "Petcare", "marque_rang": 5,
     "periodes": [(26, 8), (26, 9)], "langues": ["en", "de", "fr"], "n": 32},
]
# pic saisonnier : chocolat fondu l'été (P07, P08), toutes marques Chocolate ; ce n'est pas un incident
SAISON = {"id": "S1_fondu", "type": "saisonnier", "probleme": "fondu", "segment": "Chocolate",
          "periodes": [(25, 7), (25, 8)], "n": 45}
# campagne : même message, plusieurs marques sucrées, quelques périodes ; ce n'est pas un incident non plus
CAMPAGNE_INFO = {"id": "C1_campagne", "type": "campagne", "probleme": "additif", "segments": ["Chocolate", "Confectionery"],
                 "periodes": [(26, 3), (26, 4), (26, 5), (26, 6)], "langues": ["fr", "fr", "fr", "en", "de"],
                 "n": 40}
TAILLE_REFERENCE = 1301

# forme du fichier réel : en-têtes en ligne 3 (deux lignes vides avant), lignes vides en fin de tableau
LIGNES_AVANT_ENTETE = 2
LIGNES_VIDES_FIN = 3
COLONNES = ["Case Number", "Verbatim", "Brand", "Business Segment", "Year Period", "Blank"]


def lettre(r, langue, morceaux):
    """Met les phrases en forme de lettre, comme dans l'export : préfixe EXTERNAL, phrases souvent collées."""
    debut, fin = POLITESSE[langue]
    deja_salue = morceaux[0].split(",")[0].split("،")[0].lower() in {d.rstrip(",،").lower() for d in debut}
    if r.random() < 0.7 and not deja_salue:
        morceaux = [r.choice(debut)] + morceaux
    if r.random() < 0.7:
        morceaux = morceaux + [r.choice(fin)]
    colle = r.random() < 0.6                         # retours à la ligne perdus : phrases collées
    texte = ("" if colle else " ").join(m for m in morceaux if m)
    if r.random() < 0.06:                            # saisie négligée : tout en minuscules
        texte = texte.lower()
    return PREFIXE + texte


# détails sans rapport avec un problème précis : ils allongent la plainte sans la rendre identique aux autres
DETAILS = {
    "fr": ["J'ai dû tout jeter.", "C'est la première fois que ça m'arrive avec ce produit.", "Toute la famille l'a remarqué.", "J'ai gardé l'emballage si vous en avez besoin."],
    "en": ["I had to throw it all away.", "This is the first time this has happened with this product.", "The whole family noticed it.", "I have kept the packaging if you need it."],
    "de": ["Ich musste alles wegwerfen.", "Das ist mir mit diesem Produkt zum ersten Mal passiert.", "Die ganze Familie hat es bemerkt.", "Ich habe die Verpackung aufbewahrt, falls Sie sie brauchen."],
    "es": ["Tuve que tirarlo todo.", "Es la primera vez que me pasa con este producto.", "Toda la familia lo notó.", "He guardado el envase por si lo necesitan."],
    "it": ["Ho dovuto buttare tutto.", "È la prima volta che mi succede con questo prodotto.", "Tutta la famiglia l'ha notato.", "Ho conservato la confezione se vi serve."],
    "nl": ["Ik moest alles weggooien.", "Het is de eerste keer dat dit me overkomt met dit product.", "Het hele gezin merkte het.", "Ik heb de verpakking bewaard als u die nodig heeft."],
    "pl": ["Musiałam wszystko wyrzucić.", "To pierwszy raz, gdy zdarzyło mi się to z tym produktem.", "Cała rodzina to zauważyła.", "Zachowałam opakowanie, jeśli będzie potrzebne."],
    "ar": ["اضطررت إلى رمي كل شيء.", "هذه أول مرة يحدث لي هذا مع هذا المنتج.", "لاحظت العائلة كلها ذلك.", "احتفظت بالعبوة إذا احتجتم إليها."],
    "tr": ["Hepsini atmak zorunda kaldım.", "Bu üründe bunu ilk kez yaşıyorum.", "Bütün aile fark etti.", "İhtiyacınız olursa ambalajı sakladım."],
}


def composer(r, langue, segment, probleme):
    """Une plainte : souvent une phrase d'achat, puis surtout le problème, décrit en deux phrases ou plus."""
    t = TEXTES[langue]
    p = r.choice(t["produits"][segment])
    ouv = t["ouverture_animal"] if segment == "Petcare" and r.random() < 0.7 else t["ouverture"]
    phrases_probleme = r.sample(t[probleme], k=min(len(t[probleme]), r.choice([2, 2, 3])))
    morceaux = ([r.choice(ouv).format(p=p)] if r.random() < 0.75 else []) + phrases_probleme
    if probleme != "compliment":
        if r.random() < 0.4:
            morceaux.append(r.choice(DETAILS[langue]))
        if r.random() < 0.6:
            morceaux.append(r.choice(t["emotion"]))
        if r.random() < 0.6:
            morceaux.append(r.choice(t["demande"]))
    if not any(MASQUE_PRODUIT in m or "[PRODUCT]" in m for m in morceaux):
        morceaux.insert(0, r.choice(ouv).format(p=p))   # le produit masqué apparaît toujours, comme dans l'export
    return lettre(r, langue, morceaux)


def composer_campagne(r, langue):
    alphabet = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    lien = "https://app.notation-exemple.io/" + "".join(r.choice(alphabet) for _ in range(8))
    return lettre(r, langue, [CAMPAGNE[langue].format(m=MASQUE_PRODUIT, u=lien)])


def periode(an_rang):
    return f"{an_rang[0]:02d} P{an_rang[1]:02d}"


def generer(total, graine):
    r = random.Random(graine)
    echelle = total / TAILLE_REFERENCE
    groupes = INCIDENTS + [SAISON, CAMPAGNE_INFO]
    for g in groupes:
        g["n_effectif"] = max(5, round(g["n"] * echelle))
    n_fond = total - sum(g["n_effectif"] for g in groupes)
    marques, numero = {}, 1
    for seg, nb in SEGMENTS.items():
        marques[seg] = [f"Brand {numero + i:03d}" for i in range(nb)]
        numero += nb
    langues, poids_l = zip(*LANGUES.items())
    lignes = []

    def ajouter(verbatim, seg, marque, an_rang, etiquette):
        verbatim = verbatim.replace("[PRODUCT]", MASQUE_PRODUIT).replace("[product]", MASQUE_PRODUIT)
        lignes.append({"Verbatim": verbatim, "Brand": marque, "Business Segment": seg,
                       "Year Period": periode(an_rang), "Blank": BLANK, "_verite": etiquette})

    poids_seg = list(SEGMENTS.values())
    for _ in range(n_fond):
        seg = r.choices(list(SEGMENTS), weights=poids_seg)[0]
        langue = r.choices(langues, weights=poids_l)[0]
        probs, poids = zip(*FOND[seg].items())
        ajouter(composer(r, langue, seg, r.choices(probs, weights=poids)[0]), seg, r.choice(marques[seg]),
                r.choice(PERIODES), "")
    for inc in INCIDENTS:
        inc["marque"] = marques[inc["segment"]][inc["marque_rang"] - 1]
        for _ in range(inc["n_effectif"]):
            ajouter(composer(r, r.choice(inc["langues"]), inc["segment"], inc["probleme"]), inc["segment"],
                    inc["marque"], r.choice(inc["periodes"]), inc["id"])
    for _ in range(SAISON["n_effectif"]):
        ajouter(composer(r, r.choices(langues, weights=poids_l)[0], SAISON["segment"], SAISON["probleme"]),
                SAISON["segment"], r.choice(marques[SAISON["segment"]]), r.choice(SAISON["periodes"]), SAISON["id"])
    for _ in range(CAMPAGNE_INFO["n_effectif"]):
        seg = r.choice(CAMPAGNE_INFO["segments"])
        ajouter(composer_campagne(r, r.choice(CAMPAGNE_INFO["langues"])), seg, r.choice(marques[seg]),
                r.choice(CAMPAGNE_INFO["periodes"]), CAMPAGNE_INFO["id"])

    # numéros de cas croissants avec la période, comme un compteur de tickets
    r.shuffle(lignes)
    lignes.sort(key=lambda l: l["Year Period"])
    numeros = sorted(r.sample(range(CASE_MIN, CASE_MAX), len(lignes)))
    verite = []
    for ligne, num in zip(lignes, numeros):
        ligne["Case Number"] = num
        if ligne["_verite"]:
            verite.append({"Case Number": num, "groupe_cache": ligne["_verite"]})
    r.shuffle(lignes)
    return lignes, verite


def ecrire_table(chemin, lignes):
    """Écrit le tableau Excel : deux lignes vides, en-têtes en ligne 3, données, lignes vides à la fin."""
    from openpyxl import Workbook
    classeur = Workbook()
    feuille = classeur.active
    feuille.title = "Export"
    for _ in range(LIGNES_AVANT_ENTETE):
        feuille.append([None] * len(COLONNES))
    feuille.append(COLONNES)
    for ligne in lignes:
        feuille.append([ligne[c] for c in COLONNES])
    for _ in range(LIGNES_VIDES_FIN):
        feuille.append([None] * len(COLONNES))
    classeur.save(chemin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=TAILLE_REFERENCE, help="nombre total de plaintes")
    ap.add_argument("--graine", type=int, default=2026)
    ap.add_argument("--langues", default="", help="langues en plus de en, fr, de, séparées par des virgules (es,it,nl,pl,ar,tr)")
    ap.add_argument("--sortie", default=str(pathlib.Path(__file__).resolve().parents[1] / "donnees_fictives"))
    a = ap.parse_args()
    for l in filter(None, a.langues.split(",")):
        LANGUES[l] = POIDS_AUTRES[l]
    lignes, verite = generer(a.total, a.graine)
    sortie = pathlib.Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    ecrire_table(sortie / "plaintes_fictives.xlsx", lignes)

    # la vérité sur les groupes cachés : un seul fichier, rangé à part (il ne fait pas partie de la base)
    infos = {inc["id"]: ("incident", inc["probleme"], inc["marque"], " ".join(periode(p) for p in inc["periodes"]))
             for inc in INCIDENTS}
    infos[SAISON["id"]] = (SAISON["type"], SAISON["probleme"], "toutes " + SAISON["segment"],
                           " ".join(periode(p) for p in SAISON["periodes"]))
    infos[CAMPAGNE_INFO["id"]] = (CAMPAGNE_INFO["type"], CAMPAGNE_INFO["probleme"], "plusieurs",
                                  " ".join(periode(p) for p in CAMPAGNE_INFO["periodes"]))
    dossier_verite = sortie / "verite"
    dossier_verite.mkdir(exist_ok=True)
    with open(dossier_verite / "verite_incidents.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Case Number", "groupe_cache", "type", "probleme", "marque", "periodes"])
        for v in verite:
            w.writerow([v["Case Number"], v["groupe_cache"], *infos[v["groupe_cache"]]])
    print(f"{len(lignes)} plaintes, dont {len(verite)} dans des groupes cachés -> {sortie}")


if __name__ == "__main__":
    main()
