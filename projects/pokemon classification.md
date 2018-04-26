

```python
import requests
import PIL
import io
import numpy as np
import cv2

colors = []

page = requests.get('https://pokemondb.net/pokedex/charmander')
soup= BeautifulSoup(page.content, 'html.parser')
img_link = soup.select('div.col.desk-span-4.lap-span-6.figure img')[0]['src']
print(img_link)


img = requests.get(img_link, stream=True)
img.raw.decode_content = True
image = PIL.Image.open(img.raw)


img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#cv2.imshow('res2',res2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


```

    https://img.pokemondb.net/artwork/charmander.jpg



```python
import requests
from bs4 import BeautifulSoup
```


```python
main_page = requests.get('https://pokemondb.net/pokedex/all')
soup = BeautifulSoup(main_page.content, 'html.parser')
```


```python
poke_html_list = soup.select('a.ent-name')
poke_list = []
for poke in poke_html_list:
    if poke['href'] not in poke_list: poke_list.append(poke['href'])
```




    807




```python
pokemon_list = []
base_stats = []
type_ = []

for pokemon in poke_list:
    page2 = requests.get('https://pokemondb.net' + pokemon)
    soup2 = BeautifulSoup(page2.content, 'html.parser')
    
    #print(pokemon)
    pokemon_list.append(soup2.select('article h1')[0].text)
    
    stats = soup2.select('div.colset table.vitals-table tbody td.num') #selecting all the numbers
    stat_numbers = []
    for index, i in enumerate(stats):
        if (index)%3 ==0: stat_numbers.append(i.text) #every 3 is one of the main stats
    base_stats.append(stat_numbers)
    
    
    types = soup2.select('table.vitals-table tbody tr a.type-icon')
    ind_type = []
    for i in types:
        ind_type.append(i.text)
    type_.append(list(set(ind_type)))

```

    /pokedex/bulbasaur
    /pokedex/ivysaur
    /pokedex/venusaur
    /pokedex/charmander
    /pokedex/charmeleon
    /pokedex/charizard
    /pokedex/squirtle
    /pokedex/wartortle
    /pokedex/blastoise
    /pokedex/caterpie
    /pokedex/metapod
    /pokedex/butterfree
    /pokedex/weedle
    /pokedex/kakuna
    /pokedex/beedrill
    /pokedex/pidgey
    /pokedex/pidgeotto
    /pokedex/pidgeot
    /pokedex/rattata
    /pokedex/raticate
    /pokedex/spearow
    /pokedex/fearow
    /pokedex/ekans
    /pokedex/arbok
    /pokedex/pikachu
    /pokedex/raichu
    /pokedex/sandshrew
    /pokedex/sandslash
    /pokedex/nidoran-f
    /pokedex/nidorina
    /pokedex/nidoqueen
    /pokedex/nidoran-m
    /pokedex/nidorino
    /pokedex/nidoking
    /pokedex/clefairy
    /pokedex/clefable
    /pokedex/vulpix
    /pokedex/ninetales
    /pokedex/jigglypuff
    /pokedex/wigglytuff
    /pokedex/zubat
    /pokedex/golbat
    /pokedex/oddish
    /pokedex/gloom
    /pokedex/vileplume
    /pokedex/paras
    /pokedex/parasect
    /pokedex/venonat
    /pokedex/venomoth
    /pokedex/diglett
    /pokedex/dugtrio
    /pokedex/meowth
    /pokedex/persian
    /pokedex/psyduck
    /pokedex/golduck
    /pokedex/mankey
    /pokedex/primeape
    /pokedex/growlithe
    /pokedex/arcanine
    /pokedex/poliwag
    /pokedex/poliwhirl
    /pokedex/poliwrath
    /pokedex/abra
    /pokedex/kadabra
    /pokedex/alakazam
    /pokedex/machop
    /pokedex/machoke
    /pokedex/machamp
    /pokedex/bellsprout
    /pokedex/weepinbell
    /pokedex/victreebel
    /pokedex/tentacool
    /pokedex/tentacruel
    /pokedex/geodude
    /pokedex/graveler
    /pokedex/golem
    /pokedex/ponyta
    /pokedex/rapidash
    /pokedex/slowpoke
    /pokedex/slowbro
    /pokedex/magnemite
    /pokedex/magneton
    /pokedex/farfetchd
    /pokedex/doduo
    /pokedex/dodrio
    /pokedex/seel
    /pokedex/dewgong
    /pokedex/grimer
    /pokedex/muk
    /pokedex/shellder
    /pokedex/cloyster
    /pokedex/gastly
    /pokedex/haunter
    /pokedex/gengar
    /pokedex/onix
    /pokedex/drowzee
    /pokedex/hypno
    /pokedex/krabby
    /pokedex/kingler
    /pokedex/voltorb
    /pokedex/electrode
    /pokedex/exeggcute
    /pokedex/exeggutor
    /pokedex/cubone
    /pokedex/marowak
    /pokedex/hitmonlee
    /pokedex/hitmonchan
    /pokedex/lickitung
    /pokedex/koffing
    /pokedex/weezing
    /pokedex/rhyhorn
    /pokedex/rhydon
    /pokedex/chansey
    /pokedex/tangela
    /pokedex/kangaskhan
    /pokedex/horsea
    /pokedex/seadra
    /pokedex/goldeen
    /pokedex/seaking
    /pokedex/staryu
    /pokedex/starmie
    /pokedex/mr-mime
    /pokedex/scyther
    /pokedex/jynx
    /pokedex/electabuzz
    /pokedex/magmar
    /pokedex/pinsir
    /pokedex/tauros
    /pokedex/magikarp
    /pokedex/gyarados
    /pokedex/lapras
    /pokedex/ditto
    /pokedex/eevee
    /pokedex/vaporeon
    /pokedex/jolteon
    /pokedex/flareon
    /pokedex/porygon
    /pokedex/omanyte
    /pokedex/omastar
    /pokedex/kabuto
    /pokedex/kabutops
    /pokedex/aerodactyl
    /pokedex/snorlax
    /pokedex/articuno
    /pokedex/zapdos
    /pokedex/moltres
    /pokedex/dratini
    /pokedex/dragonair
    /pokedex/dragonite
    /pokedex/mewtwo
    /pokedex/mew
    /pokedex/chikorita
    /pokedex/bayleef
    /pokedex/meganium
    /pokedex/cyndaquil
    /pokedex/quilava
    /pokedex/typhlosion
    /pokedex/totodile
    /pokedex/croconaw
    /pokedex/feraligatr
    /pokedex/sentret
    /pokedex/furret
    /pokedex/hoothoot
    /pokedex/noctowl
    /pokedex/ledyba
    /pokedex/ledian
    /pokedex/spinarak
    /pokedex/ariados
    /pokedex/crobat
    /pokedex/chinchou
    /pokedex/lanturn
    /pokedex/pichu
    /pokedex/cleffa
    /pokedex/igglybuff
    /pokedex/togepi
    /pokedex/togetic
    /pokedex/natu
    /pokedex/xatu
    /pokedex/mareep
    /pokedex/flaaffy
    /pokedex/ampharos
    /pokedex/bellossom
    /pokedex/marill
    /pokedex/azumarill
    /pokedex/sudowoodo
    /pokedex/politoed
    /pokedex/hoppip
    /pokedex/skiploom
    /pokedex/jumpluff
    /pokedex/aipom
    /pokedex/sunkern
    /pokedex/sunflora
    /pokedex/yanma
    /pokedex/wooper
    /pokedex/quagsire
    /pokedex/espeon
    /pokedex/umbreon
    /pokedex/murkrow
    /pokedex/slowking
    /pokedex/misdreavus
    /pokedex/unown
    /pokedex/wobbuffet
    /pokedex/girafarig
    /pokedex/pineco
    /pokedex/forretress
    /pokedex/dunsparce
    /pokedex/gligar
    /pokedex/steelix
    /pokedex/snubbull
    /pokedex/granbull
    /pokedex/qwilfish
    /pokedex/scizor
    /pokedex/shuckle
    /pokedex/heracross
    /pokedex/sneasel
    /pokedex/teddiursa
    /pokedex/ursaring
    /pokedex/slugma
    /pokedex/magcargo
    /pokedex/swinub
    /pokedex/piloswine
    /pokedex/corsola
    /pokedex/remoraid
    /pokedex/octillery
    /pokedex/delibird
    /pokedex/mantine
    /pokedex/skarmory
    /pokedex/houndour
    /pokedex/houndoom
    /pokedex/kingdra
    /pokedex/phanpy
    /pokedex/donphan
    /pokedex/porygon2
    /pokedex/stantler
    /pokedex/smeargle
    /pokedex/tyrogue
    /pokedex/hitmontop
    /pokedex/smoochum
    /pokedex/elekid
    /pokedex/magby
    /pokedex/miltank
    /pokedex/blissey
    /pokedex/raikou
    /pokedex/entei
    /pokedex/suicune
    /pokedex/larvitar
    /pokedex/pupitar
    /pokedex/tyranitar
    /pokedex/lugia
    /pokedex/ho-oh
    /pokedex/celebi
    /pokedex/treecko
    /pokedex/grovyle
    /pokedex/sceptile
    /pokedex/torchic
    /pokedex/combusken
    /pokedex/blaziken
    /pokedex/mudkip
    /pokedex/marshtomp
    /pokedex/swampert
    /pokedex/poochyena
    /pokedex/mightyena
    /pokedex/zigzagoon
    /pokedex/linoone
    /pokedex/wurmple
    /pokedex/silcoon
    /pokedex/beautifly
    /pokedex/cascoon
    /pokedex/dustox
    /pokedex/lotad
    /pokedex/lombre
    /pokedex/ludicolo
    /pokedex/seedot
    /pokedex/nuzleaf
    /pokedex/shiftry
    /pokedex/taillow
    /pokedex/swellow
    /pokedex/wingull
    /pokedex/pelipper
    /pokedex/ralts
    /pokedex/kirlia
    /pokedex/gardevoir
    /pokedex/surskit
    /pokedex/masquerain
    /pokedex/shroomish
    /pokedex/breloom
    /pokedex/slakoth
    /pokedex/vigoroth
    /pokedex/slaking
    /pokedex/nincada
    /pokedex/ninjask
    /pokedex/shedinja
    /pokedex/whismur
    /pokedex/loudred
    /pokedex/exploud
    /pokedex/makuhita
    /pokedex/hariyama
    /pokedex/azurill
    /pokedex/nosepass
    /pokedex/skitty
    /pokedex/delcatty
    /pokedex/sableye
    /pokedex/mawile
    /pokedex/aron
    /pokedex/lairon
    /pokedex/aggron
    /pokedex/meditite
    /pokedex/medicham
    /pokedex/electrike
    /pokedex/manectric
    /pokedex/plusle
    /pokedex/minun
    /pokedex/volbeat
    /pokedex/illumise
    /pokedex/roselia
    /pokedex/gulpin
    /pokedex/swalot
    /pokedex/carvanha
    /pokedex/sharpedo
    /pokedex/wailmer
    /pokedex/wailord
    /pokedex/numel
    /pokedex/camerupt
    /pokedex/torkoal
    /pokedex/spoink
    /pokedex/grumpig
    /pokedex/spinda
    /pokedex/trapinch
    /pokedex/vibrava
    /pokedex/flygon
    /pokedex/cacnea
    /pokedex/cacturne
    /pokedex/swablu
    /pokedex/altaria
    /pokedex/zangoose
    /pokedex/seviper
    /pokedex/lunatone
    /pokedex/solrock
    /pokedex/barboach
    /pokedex/whiscash
    /pokedex/corphish
    /pokedex/crawdaunt
    /pokedex/baltoy
    /pokedex/claydol
    /pokedex/lileep
    /pokedex/cradily
    /pokedex/anorith
    /pokedex/armaldo
    /pokedex/feebas
    /pokedex/milotic
    /pokedex/castform
    /pokedex/kecleon
    /pokedex/shuppet
    /pokedex/banette
    /pokedex/duskull
    /pokedex/dusclops
    /pokedex/tropius
    /pokedex/chimecho
    /pokedex/absol
    /pokedex/wynaut
    /pokedex/snorunt
    /pokedex/glalie
    /pokedex/spheal
    /pokedex/sealeo
    /pokedex/walrein
    /pokedex/clamperl
    /pokedex/huntail
    /pokedex/gorebyss
    /pokedex/relicanth
    /pokedex/luvdisc
    /pokedex/bagon
    /pokedex/shelgon
    /pokedex/salamence
    /pokedex/beldum
    /pokedex/metang
    /pokedex/metagross
    /pokedex/regirock
    /pokedex/regice
    /pokedex/registeel
    /pokedex/latias
    /pokedex/latios
    /pokedex/kyogre
    /pokedex/groudon
    /pokedex/rayquaza
    /pokedex/jirachi
    /pokedex/deoxys
    /pokedex/turtwig
    /pokedex/grotle
    /pokedex/torterra
    /pokedex/chimchar
    /pokedex/monferno
    /pokedex/infernape
    /pokedex/piplup
    /pokedex/prinplup
    /pokedex/empoleon
    /pokedex/starly
    /pokedex/staravia
    /pokedex/staraptor
    /pokedex/bidoof
    /pokedex/bibarel
    /pokedex/kricketot
    /pokedex/kricketune
    /pokedex/shinx
    /pokedex/luxio
    /pokedex/luxray
    /pokedex/budew
    /pokedex/roserade
    /pokedex/cranidos
    /pokedex/rampardos
    /pokedex/shieldon
    /pokedex/bastiodon
    /pokedex/burmy
    /pokedex/wormadam
    /pokedex/mothim
    /pokedex/combee
    /pokedex/vespiquen
    /pokedex/pachirisu
    /pokedex/buizel
    /pokedex/floatzel
    /pokedex/cherubi
    /pokedex/cherrim
    /pokedex/shellos
    /pokedex/gastrodon
    /pokedex/ambipom
    /pokedex/drifloon
    /pokedex/drifblim
    /pokedex/buneary
    /pokedex/lopunny
    /pokedex/mismagius
    /pokedex/honchkrow
    /pokedex/glameow
    /pokedex/purugly
    /pokedex/chingling
    /pokedex/stunky
    /pokedex/skuntank
    /pokedex/bronzor
    /pokedex/bronzong
    /pokedex/bonsly
    /pokedex/mime-jr
    /pokedex/happiny
    /pokedex/chatot
    /pokedex/spiritomb
    /pokedex/gible
    /pokedex/gabite
    /pokedex/garchomp
    /pokedex/munchlax
    /pokedex/riolu
    /pokedex/lucario
    /pokedex/hippopotas
    /pokedex/hippowdon
    /pokedex/skorupi
    /pokedex/drapion
    /pokedex/croagunk
    /pokedex/toxicroak
    /pokedex/carnivine
    /pokedex/finneon
    /pokedex/lumineon
    /pokedex/mantyke
    /pokedex/snover
    /pokedex/abomasnow
    /pokedex/weavile
    /pokedex/magnezone
    /pokedex/lickilicky
    /pokedex/rhyperior
    /pokedex/tangrowth
    /pokedex/electivire
    /pokedex/magmortar
    /pokedex/togekiss
    /pokedex/yanmega
    /pokedex/leafeon
    /pokedex/glaceon
    /pokedex/gliscor
    /pokedex/mamoswine
    /pokedex/porygon-z
    /pokedex/gallade
    /pokedex/probopass
    /pokedex/dusknoir
    /pokedex/froslass
    /pokedex/rotom
    /pokedex/uxie
    /pokedex/mesprit
    /pokedex/azelf
    /pokedex/dialga
    /pokedex/palkia
    /pokedex/heatran
    /pokedex/regigigas
    /pokedex/giratina
    /pokedex/cresselia
    /pokedex/phione
    /pokedex/manaphy
    /pokedex/darkrai
    /pokedex/shaymin
    /pokedex/arceus
    /pokedex/victini
    /pokedex/snivy
    /pokedex/servine
    /pokedex/serperior
    /pokedex/tepig
    /pokedex/pignite
    /pokedex/emboar
    /pokedex/oshawott
    /pokedex/dewott
    /pokedex/samurott
    /pokedex/patrat
    /pokedex/watchog
    /pokedex/lillipup
    /pokedex/herdier
    /pokedex/stoutland
    /pokedex/purrloin
    /pokedex/liepard
    /pokedex/pansage
    /pokedex/simisage
    /pokedex/pansear
    /pokedex/simisear
    /pokedex/panpour
    /pokedex/simipour
    /pokedex/munna
    /pokedex/musharna
    /pokedex/pidove
    /pokedex/tranquill
    /pokedex/unfezant
    /pokedex/blitzle
    /pokedex/zebstrika
    /pokedex/roggenrola
    /pokedex/boldore
    /pokedex/gigalith
    /pokedex/woobat
    /pokedex/swoobat
    /pokedex/drilbur
    /pokedex/excadrill
    /pokedex/audino
    /pokedex/timburr
    /pokedex/gurdurr
    /pokedex/conkeldurr
    /pokedex/tympole
    /pokedex/palpitoad
    /pokedex/seismitoad
    /pokedex/throh
    /pokedex/sawk
    /pokedex/sewaddle
    /pokedex/swadloon
    /pokedex/leavanny
    /pokedex/venipede
    /pokedex/whirlipede
    /pokedex/scolipede
    /pokedex/cottonee
    /pokedex/whimsicott
    /pokedex/petilil
    /pokedex/lilligant
    /pokedex/basculin
    /pokedex/sandile
    /pokedex/krokorok
    /pokedex/krookodile
    /pokedex/darumaka
    /pokedex/darmanitan
    /pokedex/maractus
    /pokedex/dwebble
    /pokedex/crustle
    /pokedex/scraggy
    /pokedex/scrafty
    /pokedex/sigilyph
    /pokedex/yamask
    /pokedex/cofagrigus
    /pokedex/tirtouga
    /pokedex/carracosta
    /pokedex/archen
    /pokedex/archeops
    /pokedex/trubbish
    /pokedex/garbodor
    /pokedex/zorua
    /pokedex/zoroark
    /pokedex/minccino
    /pokedex/cinccino
    /pokedex/gothita
    /pokedex/gothorita
    /pokedex/gothitelle
    /pokedex/solosis
    /pokedex/duosion
    /pokedex/reuniclus
    /pokedex/ducklett
    /pokedex/swanna
    /pokedex/vanillite
    /pokedex/vanillish
    /pokedex/vanilluxe
    /pokedex/deerling
    /pokedex/sawsbuck
    /pokedex/emolga
    /pokedex/karrablast
    /pokedex/escavalier
    /pokedex/foongus
    /pokedex/amoonguss
    /pokedex/frillish
    /pokedex/jellicent
    /pokedex/alomomola
    /pokedex/joltik
    /pokedex/galvantula
    /pokedex/ferroseed
    /pokedex/ferrothorn
    /pokedex/klink
    /pokedex/klang
    /pokedex/klinklang
    /pokedex/tynamo
    /pokedex/eelektrik
    /pokedex/eelektross
    /pokedex/elgyem
    /pokedex/beheeyem
    /pokedex/litwick
    /pokedex/lampent
    /pokedex/chandelure
    /pokedex/axew
    /pokedex/fraxure
    /pokedex/haxorus
    /pokedex/cubchoo
    /pokedex/beartic
    /pokedex/cryogonal
    /pokedex/shelmet
    /pokedex/accelgor
    /pokedex/stunfisk
    /pokedex/mienfoo
    /pokedex/mienshao
    /pokedex/druddigon
    /pokedex/golett
    /pokedex/golurk
    /pokedex/pawniard
    /pokedex/bisharp
    /pokedex/bouffalant
    /pokedex/rufflet
    /pokedex/braviary
    /pokedex/vullaby
    /pokedex/mandibuzz
    /pokedex/heatmor
    /pokedex/durant
    /pokedex/deino
    /pokedex/zweilous
    /pokedex/hydreigon
    /pokedex/larvesta
    /pokedex/volcarona
    /pokedex/cobalion
    /pokedex/terrakion
    /pokedex/virizion
    /pokedex/tornadus
    /pokedex/thundurus
    /pokedex/reshiram
    /pokedex/zekrom
    /pokedex/landorus
    /pokedex/kyurem
    /pokedex/keldeo
    /pokedex/meloetta
    /pokedex/genesect
    /pokedex/chespin
    /pokedex/quilladin
    /pokedex/chesnaught
    /pokedex/fennekin
    /pokedex/braixen
    /pokedex/delphox
    /pokedex/froakie
    /pokedex/frogadier
    /pokedex/greninja
    /pokedex/bunnelby
    /pokedex/diggersby
    /pokedex/fletchling
    /pokedex/fletchinder
    /pokedex/talonflame
    /pokedex/scatterbug
    /pokedex/spewpa
    /pokedex/vivillon
    /pokedex/litleo
    /pokedex/pyroar
    /pokedex/flabebe
    /pokedex/floette
    /pokedex/florges
    /pokedex/skiddo
    /pokedex/gogoat
    /pokedex/pancham
    /pokedex/pangoro
    /pokedex/furfrou
    /pokedex/espurr
    /pokedex/meowstic
    /pokedex/honedge
    /pokedex/doublade
    /pokedex/aegislash
    /pokedex/spritzee
    /pokedex/aromatisse
    /pokedex/swirlix
    /pokedex/slurpuff
    /pokedex/inkay
    /pokedex/malamar
    /pokedex/binacle
    /pokedex/barbaracle
    /pokedex/skrelp
    /pokedex/dragalge
    /pokedex/clauncher
    /pokedex/clawitzer
    /pokedex/helioptile
    /pokedex/heliolisk
    /pokedex/tyrunt
    /pokedex/tyrantrum
    /pokedex/amaura
    /pokedex/aurorus
    /pokedex/sylveon
    /pokedex/hawlucha
    /pokedex/dedenne
    /pokedex/carbink
    /pokedex/goomy
    /pokedex/sliggoo
    /pokedex/goodra
    /pokedex/klefki
    /pokedex/phantump
    /pokedex/trevenant
    /pokedex/pumpkaboo
    /pokedex/gourgeist
    /pokedex/bergmite
    /pokedex/avalugg
    /pokedex/noibat
    /pokedex/noivern
    /pokedex/xerneas
    /pokedex/yveltal
    /pokedex/zygarde
    /pokedex/diancie
    /pokedex/hoopa
    /pokedex/volcanion
    /pokedex/rowlet
    /pokedex/dartrix
    /pokedex/decidueye
    /pokedex/litten
    /pokedex/torracat
    /pokedex/incineroar
    /pokedex/popplio
    /pokedex/brionne
    /pokedex/primarina
    /pokedex/pikipek
    /pokedex/trumbeak
    /pokedex/toucannon
    /pokedex/yungoos
    /pokedex/gumshoos
    /pokedex/grubbin
    /pokedex/charjabug
    /pokedex/vikavolt
    /pokedex/crabrawler
    /pokedex/crabominable
    /pokedex/oricorio
    /pokedex/cutiefly
    /pokedex/ribombee
    /pokedex/rockruff
    /pokedex/lycanroc
    /pokedex/wishiwashi
    /pokedex/mareanie
    /pokedex/toxapex
    /pokedex/mudbray
    /pokedex/mudsdale
    /pokedex/dewpider
    /pokedex/araquanid
    /pokedex/fomantis
    /pokedex/lurantis
    /pokedex/morelull
    /pokedex/shiinotic
    /pokedex/salandit
    /pokedex/salazzle
    /pokedex/stufful
    /pokedex/bewear
    /pokedex/bounsweet
    /pokedex/steenee
    /pokedex/tsareena
    /pokedex/comfey
    /pokedex/oranguru
    /pokedex/passimian
    /pokedex/wimpod
    /pokedex/golisopod
    /pokedex/sandygast
    /pokedex/palossand
    /pokedex/pyukumuku
    /pokedex/type-null
    /pokedex/silvally
    /pokedex/minior
    /pokedex/komala
    /pokedex/turtonator
    /pokedex/togedemaru
    /pokedex/mimikyu
    /pokedex/bruxish
    /pokedex/drampa
    /pokedex/dhelmise
    /pokedex/jangmo-o
    /pokedex/hakamo-o
    /pokedex/kommo-o
    /pokedex/tapu-koko
    /pokedex/tapu-lele
    /pokedex/tapu-bulu
    /pokedex/tapu-fini
    /pokedex/cosmog
    /pokedex/cosmoem
    /pokedex/solgaleo
    /pokedex/lunala
    /pokedex/nihilego
    /pokedex/buzzwole
    /pokedex/pheromosa
    /pokedex/xurkitree
    /pokedex/celesteela
    /pokedex/kartana
    /pokedex/guzzlord
    /pokedex/necrozma
    /pokedex/magearna
    /pokedex/marshadow
    /pokedex/poipole
    /pokedex/naganadel
    /pokedex/stakataka
    /pokedex/blacephalon
    /pokedex/zeraora



```python
hp = []
att = []
defs = []
spatt = []
spdef = []
spe = []

for i in base_stats:
    hp.append(i[0])
    att.append(i[1])
    defs.append(i[2])
    spatt.append(i[3])
    spdef.append(i[4])
    spe.append(i[5])
    
first_type = []
for i in type_:
    first_type.append(i[0])
```


```python
import pandas as pd
dataf = pd.DataFrame({'pokemon': pokemon_list,
                      'hp': hp,
                      'att': att,
                      'defs': defs,
                      'spatt': spatt,
                      'spdef': spdef,
                      'spe': spe,
                      'type': first_type,
                      'types': type_})


# Setting up the training and testing datasets
dataf2 = dataf.copy().drop('pokemon', axis=1).as_matrix()

train = dataf2[0:721]
trainX = train[:,0:5]
trainy = train[:,6]
test = dataf2[721:len(dataf), ]
testX = test[:,0:5]
testy = test[:,6]
testy2 = test[:,7]

```


```python
from sklearn import svm
svm_clf = svm.SVC(gamma = .00001, C = 1000)
svm_clf.fit(trainX, trainy)
print('svm class rate:', sum(svm_clf.predict(testX) == testy)/len(testy))

from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier(n_neighbors = 2)
knn_clf.fit(trainX, trainy)
print('knn classification rate:', str(sum(knn_clf.predict(testX) == testy)/len(testy)))

```

    svm class rate: 0.13953488372093023
    knn classification rate: 0.12790697674418605



```python
two_colors = []

big_list = []
res3 = res2.tolist()

for i in res3:
    big_list.extend(i)
for i in big_list:
    in_list = i in two_colors
    if in_list == False: two_colors.append(i)

```


```python
import requests
import PIL
import io
import numpy as np
import cv2

poke_col_list = []

for pokemon in poke_list:
    page2 = requests.get('https://pokemondb.net' + pokemon)
    soup2 = BeautifulSoup(page2.content, 'html.parser')

    img_link = soup2.select('div.col.desk-span-4.lap-span-6.figure img')[0]['src']
    print(img_link)

    img = requests.get(img_link, stream=True)
    img.raw.decode_content = True
    image = PIL.Image.open(img.raw)


    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    # extracting the two colors
    two_colors = []
    big_list = []
    res3 = res2.tolist()
    for i in res3:
        big_list.extend(i)
    for i in big_list:
        in_list = i in two_colors
        if in_list == False: two_colors.append(i)
    # the color that is farther away from 250 (white) is probably the pokemon's dominant color
    
    main_color = [sorted(two_colors)[0]] # each array is (BGR) instead of (RGB)
    poke_col_list.append(main_color)
    
```

    https://img.pokemondb.net/artwork/bulbasaur.jpg
    https://img.pokemondb.net/artwork/ivysaur.jpg
    https://img.pokemondb.net/artwork/venusaur.jpg
    https://img.pokemondb.net/artwork/charmander.jpg
    https://img.pokemondb.net/artwork/charmeleon.jpg
    https://img.pokemondb.net/artwork/charizard.jpg
    https://img.pokemondb.net/artwork/squirtle.jpg
    https://img.pokemondb.net/artwork/wartortle.jpg
    https://img.pokemondb.net/artwork/blastoise.jpg
    https://img.pokemondb.net/artwork/caterpie.jpg
    https://img.pokemondb.net/artwork/metapod.jpg
    https://img.pokemondb.net/artwork/butterfree.jpg
    https://img.pokemondb.net/artwork/weedle.jpg
    https://img.pokemondb.net/artwork/kakuna.jpg
    https://img.pokemondb.net/artwork/beedrill.jpg
    https://img.pokemondb.net/artwork/pidgey.jpg
    https://img.pokemondb.net/artwork/pidgeotto.jpg
    https://img.pokemondb.net/artwork/pidgeot.jpg
    https://img.pokemondb.net/artwork/rattata.jpg
    https://img.pokemondb.net/artwork/raticate.jpg
    https://img.pokemondb.net/artwork/spearow.jpg
    https://img.pokemondb.net/artwork/fearow.jpg
    https://img.pokemondb.net/artwork/ekans.jpg
    https://img.pokemondb.net/artwork/arbok.jpg
    https://img.pokemondb.net/artwork/pikachu.jpg
    https://img.pokemondb.net/artwork/raichu.jpg
    https://img.pokemondb.net/artwork/sandshrew.jpg
    https://img.pokemondb.net/artwork/sandslash.jpg
    https://img.pokemondb.net/artwork/nidoran-f.jpg
    https://img.pokemondb.net/artwork/nidorina.jpg
    https://img.pokemondb.net/artwork/nidoqueen.jpg
    https://img.pokemondb.net/artwork/nidoran-m.jpg
    https://img.pokemondb.net/artwork/nidorino.jpg
    https://img.pokemondb.net/artwork/nidoking.jpg
    https://img.pokemondb.net/artwork/clefairy.jpg
    https://img.pokemondb.net/artwork/clefable.jpg
    https://img.pokemondb.net/artwork/vulpix.jpg
    https://img.pokemondb.net/artwork/ninetales.jpg
    https://img.pokemondb.net/artwork/jigglypuff.jpg
    https://img.pokemondb.net/artwork/wigglytuff.jpg
    https://img.pokemondb.net/artwork/zubat.jpg
    https://img.pokemondb.net/artwork/golbat.jpg
    https://img.pokemondb.net/artwork/oddish.jpg
    https://img.pokemondb.net/artwork/gloom.jpg
    https://img.pokemondb.net/artwork/vileplume.jpg
    https://img.pokemondb.net/artwork/paras.jpg
    https://img.pokemondb.net/artwork/parasect.jpg
    https://img.pokemondb.net/artwork/venonat.jpg
    https://img.pokemondb.net/artwork/venomoth.jpg
    https://img.pokemondb.net/artwork/diglett.jpg
    https://img.pokemondb.net/artwork/dugtrio.jpg
    https://img.pokemondb.net/artwork/meowth.jpg
    https://img.pokemondb.net/artwork/persian.jpg
    https://img.pokemondb.net/artwork/psyduck.jpg
    https://img.pokemondb.net/artwork/golduck.jpg
    https://img.pokemondb.net/artwork/mankey.jpg
    https://img.pokemondb.net/artwork/primeape.jpg
    https://img.pokemondb.net/artwork/growlithe.jpg
    https://img.pokemondb.net/artwork/arcanine.jpg
    https://img.pokemondb.net/artwork/poliwag.jpg
    https://img.pokemondb.net/artwork/poliwhirl.jpg
    https://img.pokemondb.net/artwork/poliwrath.jpg
    https://img.pokemondb.net/artwork/abra.jpg
    https://img.pokemondb.net/artwork/kadabra.jpg
    https://img.pokemondb.net/artwork/alakazam.jpg
    https://img.pokemondb.net/artwork/machop.jpg
    https://img.pokemondb.net/artwork/machoke.jpg
    https://img.pokemondb.net/artwork/machamp.jpg
    https://img.pokemondb.net/artwork/bellsprout.jpg
    https://img.pokemondb.net/artwork/weepinbell.jpg
    https://img.pokemondb.net/artwork/victreebel.jpg
    https://img.pokemondb.net/artwork/tentacool.jpg
    https://img.pokemondb.net/artwork/tentacruel.jpg
    https://img.pokemondb.net/artwork/geodude.jpg
    https://img.pokemondb.net/artwork/graveler.jpg
    https://img.pokemondb.net/artwork/golem.jpg
    https://img.pokemondb.net/artwork/ponyta.jpg
    https://img.pokemondb.net/artwork/rapidash.jpg
    https://img.pokemondb.net/artwork/slowpoke.jpg
    https://img.pokemondb.net/artwork/slowbro.jpg
    https://img.pokemondb.net/artwork/magnemite.jpg
    https://img.pokemondb.net/artwork/magneton.jpg
    https://img.pokemondb.net/artwork/farfetchd.jpg
    https://img.pokemondb.net/artwork/doduo.jpg
    https://img.pokemondb.net/artwork/dodrio.jpg
    https://img.pokemondb.net/artwork/seel.jpg
    https://img.pokemondb.net/artwork/dewgong.jpg
    https://img.pokemondb.net/artwork/grimer.jpg
    https://img.pokemondb.net/artwork/muk.jpg
    https://img.pokemondb.net/artwork/shellder.jpg
    https://img.pokemondb.net/artwork/cloyster.jpg
    https://img.pokemondb.net/artwork/gastly.jpg
    https://img.pokemondb.net/artwork/haunter.jpg
    https://img.pokemondb.net/artwork/gengar.jpg
    https://img.pokemondb.net/artwork/onix.jpg
    https://img.pokemondb.net/artwork/drowzee.jpg
    https://img.pokemondb.net/artwork/hypno.jpg
    https://img.pokemondb.net/artwork/krabby.jpg
    https://img.pokemondb.net/artwork/kingler.jpg
    https://img.pokemondb.net/artwork/voltorb.jpg
    https://img.pokemondb.net/artwork/electrode.jpg
    https://img.pokemondb.net/artwork/exeggcute.jpg
    https://img.pokemondb.net/artwork/exeggutor.jpg
    https://img.pokemondb.net/artwork/cubone.jpg
    https://img.pokemondb.net/artwork/marowak.jpg
    https://img.pokemondb.net/artwork/hitmonlee.jpg
    https://img.pokemondb.net/artwork/hitmonchan.jpg
    https://img.pokemondb.net/artwork/lickitung.jpg
    https://img.pokemondb.net/artwork/koffing.jpg
    https://img.pokemondb.net/artwork/weezing.jpg
    https://img.pokemondb.net/artwork/rhyhorn.jpg
    https://img.pokemondb.net/artwork/rhydon.jpg
    https://img.pokemondb.net/artwork/chansey.jpg
    https://img.pokemondb.net/artwork/tangela.jpg
    https://img.pokemondb.net/artwork/kangaskhan.jpg
    https://img.pokemondb.net/artwork/horsea.jpg
    https://img.pokemondb.net/artwork/seadra.jpg
    https://img.pokemondb.net/artwork/goldeen.jpg
    https://img.pokemondb.net/artwork/seaking.jpg
    https://img.pokemondb.net/artwork/staryu.jpg
    https://img.pokemondb.net/artwork/starmie.jpg
    https://img.pokemondb.net/artwork/mr-mime.jpg
    https://img.pokemondb.net/artwork/scyther.jpg
    https://img.pokemondb.net/artwork/jynx.jpg
    https://img.pokemondb.net/artwork/electabuzz.jpg
    https://img.pokemondb.net/artwork/magmar.jpg
    https://img.pokemondb.net/artwork/pinsir.jpg
    https://img.pokemondb.net/artwork/tauros.jpg
    https://img.pokemondb.net/artwork/magikarp.jpg
    https://img.pokemondb.net/artwork/gyarados.jpg
    https://img.pokemondb.net/artwork/lapras.jpg
    https://img.pokemondb.net/artwork/ditto.jpg
    https://img.pokemondb.net/artwork/eevee.jpg
    https://img.pokemondb.net/artwork/vaporeon.jpg
    https://img.pokemondb.net/artwork/jolteon.jpg
    https://img.pokemondb.net/artwork/flareon.jpg
    https://img.pokemondb.net/artwork/porygon.jpg
    https://img.pokemondb.net/artwork/omanyte.jpg
    https://img.pokemondb.net/artwork/omastar.jpg
    https://img.pokemondb.net/artwork/kabuto.jpg
    https://img.pokemondb.net/artwork/kabutops.jpg
    https://img.pokemondb.net/artwork/aerodactyl.jpg
    https://img.pokemondb.net/artwork/snorlax.jpg
    https://img.pokemondb.net/artwork/articuno.jpg
    https://img.pokemondb.net/artwork/zapdos.jpg
    https://img.pokemondb.net/artwork/moltres.jpg
    https://img.pokemondb.net/artwork/dratini.jpg
    https://img.pokemondb.net/artwork/dragonair.jpg
    https://img.pokemondb.net/artwork/dragonite.jpg
    https://img.pokemondb.net/artwork/mewtwo.jpg
    https://img.pokemondb.net/artwork/mew.jpg
    https://img.pokemondb.net/artwork/chikorita.jpg
    https://img.pokemondb.net/artwork/bayleef.jpg
    https://img.pokemondb.net/artwork/meganium.jpg
    https://img.pokemondb.net/artwork/cyndaquil.jpg
    https://img.pokemondb.net/artwork/quilava.jpg
    https://img.pokemondb.net/artwork/typhlosion.jpg
    https://img.pokemondb.net/artwork/totodile.jpg
    https://img.pokemondb.net/artwork/croconaw.jpg
    https://img.pokemondb.net/artwork/feraligatr.jpg
    https://img.pokemondb.net/artwork/sentret.jpg
    https://img.pokemondb.net/artwork/furret.jpg
    https://img.pokemondb.net/artwork/hoothoot.jpg
    https://img.pokemondb.net/artwork/noctowl.jpg
    https://img.pokemondb.net/artwork/ledyba.jpg
    https://img.pokemondb.net/artwork/ledian.jpg
    https://img.pokemondb.net/artwork/spinarak.jpg
    https://img.pokemondb.net/artwork/ariados.jpg
    https://img.pokemondb.net/artwork/crobat.jpg
    https://img.pokemondb.net/artwork/chinchou.jpg
    https://img.pokemondb.net/artwork/lanturn.jpg
    https://img.pokemondb.net/artwork/pichu.jpg
    https://img.pokemondb.net/artwork/cleffa.jpg
    https://img.pokemondb.net/artwork/igglybuff.jpg
    https://img.pokemondb.net/artwork/togepi.jpg
    https://img.pokemondb.net/artwork/togetic.jpg
    https://img.pokemondb.net/artwork/natu.jpg
    https://img.pokemondb.net/artwork/xatu.jpg
    https://img.pokemondb.net/artwork/mareep.jpg
    https://img.pokemondb.net/artwork/flaaffy.jpg
    https://img.pokemondb.net/artwork/ampharos.jpg
    https://img.pokemondb.net/artwork/bellossom.jpg
    https://img.pokemondb.net/artwork/marill.jpg
    https://img.pokemondb.net/artwork/azumarill.jpg
    https://img.pokemondb.net/artwork/sudowoodo.jpg
    https://img.pokemondb.net/artwork/politoed.jpg
    https://img.pokemondb.net/artwork/hoppip.jpg
    https://img.pokemondb.net/artwork/skiploom.jpg
    https://img.pokemondb.net/artwork/jumpluff.jpg
    https://img.pokemondb.net/artwork/aipom.jpg
    https://img.pokemondb.net/artwork/sunkern.jpg
    https://img.pokemondb.net/artwork/sunflora.jpg
    https://img.pokemondb.net/artwork/yanma.jpg
    https://img.pokemondb.net/artwork/wooper.jpg
    https://img.pokemondb.net/artwork/quagsire.jpg
    https://img.pokemondb.net/artwork/espeon.jpg
    https://img.pokemondb.net/artwork/umbreon.jpg
    https://img.pokemondb.net/artwork/murkrow.jpg
    https://img.pokemondb.net/artwork/slowking.jpg
    https://img.pokemondb.net/artwork/misdreavus.jpg
    https://img.pokemondb.net/artwork/unown.jpg
    https://img.pokemondb.net/artwork/wobbuffet.jpg
    https://img.pokemondb.net/artwork/girafarig.jpg
    https://img.pokemondb.net/artwork/pineco.jpg
    https://img.pokemondb.net/artwork/forretress.jpg
    https://img.pokemondb.net/artwork/dunsparce.jpg
    https://img.pokemondb.net/artwork/gligar.jpg
    https://img.pokemondb.net/artwork/steelix.jpg
    https://img.pokemondb.net/artwork/snubbull.jpg
    https://img.pokemondb.net/artwork/granbull.jpg
    https://img.pokemondb.net/artwork/qwilfish.jpg
    https://img.pokemondb.net/artwork/scizor.jpg
    https://img.pokemondb.net/artwork/shuckle.jpg
    https://img.pokemondb.net/artwork/heracross.jpg
    https://img.pokemondb.net/artwork/sneasel.jpg
    https://img.pokemondb.net/artwork/teddiursa.jpg
    https://img.pokemondb.net/artwork/ursaring.jpg
    https://img.pokemondb.net/artwork/slugma.jpg
    https://img.pokemondb.net/artwork/magcargo.jpg
    https://img.pokemondb.net/artwork/swinub.jpg
    https://img.pokemondb.net/artwork/piloswine.jpg
    https://img.pokemondb.net/artwork/corsola.jpg
    https://img.pokemondb.net/artwork/remoraid.jpg
    https://img.pokemondb.net/artwork/octillery.jpg
    https://img.pokemondb.net/artwork/delibird.jpg
    https://img.pokemondb.net/artwork/mantine.jpg
    https://img.pokemondb.net/artwork/skarmory.jpg
    https://img.pokemondb.net/artwork/houndour.jpg
    https://img.pokemondb.net/artwork/houndoom.jpg
    https://img.pokemondb.net/artwork/kingdra.jpg
    https://img.pokemondb.net/artwork/phanpy.jpg
    https://img.pokemondb.net/artwork/donphan.jpg
    https://img.pokemondb.net/artwork/porygon2.jpg
    https://img.pokemondb.net/artwork/stantler.jpg
    https://img.pokemondb.net/artwork/smeargle.jpg
    https://img.pokemondb.net/artwork/tyrogue.jpg
    https://img.pokemondb.net/artwork/hitmontop.jpg
    https://img.pokemondb.net/artwork/smoochum.jpg
    https://img.pokemondb.net/artwork/elekid.jpg
    https://img.pokemondb.net/artwork/magby.jpg
    https://img.pokemondb.net/artwork/miltank.jpg
    https://img.pokemondb.net/artwork/blissey.jpg
    https://img.pokemondb.net/artwork/raikou.jpg
    https://img.pokemondb.net/artwork/entei.jpg
    https://img.pokemondb.net/artwork/suicune.jpg
    https://img.pokemondb.net/artwork/larvitar.jpg
    https://img.pokemondb.net/artwork/pupitar.jpg
    https://img.pokemondb.net/artwork/tyranitar.jpg
    https://img.pokemondb.net/artwork/lugia.jpg
    https://img.pokemondb.net/artwork/ho-oh.jpg
    https://img.pokemondb.net/artwork/celebi.jpg
    https://img.pokemondb.net/artwork/treecko.jpg
    https://img.pokemondb.net/artwork/grovyle.jpg
    https://img.pokemondb.net/artwork/sceptile.jpg
    https://img.pokemondb.net/artwork/torchic.jpg
    https://img.pokemondb.net/artwork/combusken.jpg
    https://img.pokemondb.net/artwork/blaziken.jpg
    https://img.pokemondb.net/artwork/mudkip.jpg
    https://img.pokemondb.net/artwork/marshtomp.jpg
    https://img.pokemondb.net/artwork/swampert.jpg
    https://img.pokemondb.net/artwork/poochyena.jpg
    https://img.pokemondb.net/artwork/mightyena.jpg
    https://img.pokemondb.net/artwork/zigzagoon.jpg
    https://img.pokemondb.net/artwork/linoone.jpg
    https://img.pokemondb.net/artwork/wurmple.jpg
    https://img.pokemondb.net/artwork/silcoon.jpg
    https://img.pokemondb.net/artwork/beautifly.jpg
    https://img.pokemondb.net/artwork/cascoon.jpg
    https://img.pokemondb.net/artwork/dustox.jpg
    https://img.pokemondb.net/artwork/lotad.jpg
    https://img.pokemondb.net/artwork/lombre.jpg
    https://img.pokemondb.net/artwork/ludicolo.jpg
    https://img.pokemondb.net/artwork/seedot.jpg
    https://img.pokemondb.net/artwork/nuzleaf.jpg
    https://img.pokemondb.net/artwork/shiftry.jpg
    https://img.pokemondb.net/artwork/taillow.jpg
    https://img.pokemondb.net/artwork/swellow.jpg
    https://img.pokemondb.net/artwork/wingull.jpg
    https://img.pokemondb.net/artwork/pelipper.jpg
    https://img.pokemondb.net/artwork/ralts.jpg
    https://img.pokemondb.net/artwork/kirlia.jpg
    https://img.pokemondb.net/artwork/gardevoir.jpg
    https://img.pokemondb.net/artwork/surskit.jpg
    https://img.pokemondb.net/artwork/masquerain.jpg
    https://img.pokemondb.net/artwork/shroomish.jpg
    https://img.pokemondb.net/artwork/breloom.jpg
    https://img.pokemondb.net/artwork/slakoth.jpg
    https://img.pokemondb.net/artwork/vigoroth.jpg
    https://img.pokemondb.net/artwork/slaking.jpg
    https://img.pokemondb.net/artwork/nincada.jpg
    https://img.pokemondb.net/artwork/ninjask.jpg
    https://img.pokemondb.net/artwork/shedinja.jpg
    https://img.pokemondb.net/artwork/whismur.jpg
    https://img.pokemondb.net/artwork/loudred.jpg
    https://img.pokemondb.net/artwork/exploud.jpg
    https://img.pokemondb.net/artwork/makuhita.jpg
    https://img.pokemondb.net/artwork/hariyama.jpg
    https://img.pokemondb.net/artwork/azurill.jpg
    https://img.pokemondb.net/artwork/nosepass.jpg
    https://img.pokemondb.net/artwork/skitty.jpg
    https://img.pokemondb.net/artwork/delcatty.jpg
    https://img.pokemondb.net/artwork/sableye.jpg
    https://img.pokemondb.net/artwork/mawile.jpg
    https://img.pokemondb.net/artwork/aron.jpg
    https://img.pokemondb.net/artwork/lairon.jpg
    https://img.pokemondb.net/artwork/aggron.jpg
    https://img.pokemondb.net/artwork/meditite.jpg
    https://img.pokemondb.net/artwork/medicham.jpg
    https://img.pokemondb.net/artwork/electrike.jpg
    https://img.pokemondb.net/artwork/manectric.jpg
    https://img.pokemondb.net/artwork/plusle.jpg
    https://img.pokemondb.net/artwork/minun.jpg
    https://img.pokemondb.net/artwork/volbeat.jpg
    https://img.pokemondb.net/artwork/illumise.jpg
    https://img.pokemondb.net/artwork/roselia.jpg
    https://img.pokemondb.net/artwork/gulpin.jpg
    https://img.pokemondb.net/artwork/swalot.jpg
    https://img.pokemondb.net/artwork/carvanha.jpg
    https://img.pokemondb.net/artwork/sharpedo.jpg
    https://img.pokemondb.net/artwork/wailmer.jpg
    https://img.pokemondb.net/artwork/wailord.jpg
    https://img.pokemondb.net/artwork/numel.jpg
    https://img.pokemondb.net/artwork/camerupt.jpg
    https://img.pokemondb.net/artwork/torkoal.jpg
    https://img.pokemondb.net/artwork/spoink.jpg
    https://img.pokemondb.net/artwork/grumpig.jpg
    https://img.pokemondb.net/artwork/spinda.jpg
    https://img.pokemondb.net/artwork/trapinch.jpg
    https://img.pokemondb.net/artwork/vibrava.jpg
    https://img.pokemondb.net/artwork/flygon.jpg
    https://img.pokemondb.net/artwork/cacnea.jpg
    https://img.pokemondb.net/artwork/cacturne.jpg
    https://img.pokemondb.net/artwork/swablu.jpg
    https://img.pokemondb.net/artwork/altaria.jpg
    https://img.pokemondb.net/artwork/zangoose.jpg
    https://img.pokemondb.net/artwork/seviper.jpg
    https://img.pokemondb.net/artwork/lunatone.jpg
    https://img.pokemondb.net/artwork/solrock.jpg
    https://img.pokemondb.net/artwork/barboach.jpg
    https://img.pokemondb.net/artwork/whiscash.jpg
    https://img.pokemondb.net/artwork/corphish.jpg
    https://img.pokemondb.net/artwork/crawdaunt.jpg
    https://img.pokemondb.net/artwork/baltoy.jpg
    https://img.pokemondb.net/artwork/claydol.jpg
    https://img.pokemondb.net/artwork/lileep.jpg
    https://img.pokemondb.net/artwork/cradily.jpg
    https://img.pokemondb.net/artwork/anorith.jpg
    https://img.pokemondb.net/artwork/armaldo.jpg
    https://img.pokemondb.net/artwork/feebas.jpg
    https://img.pokemondb.net/artwork/milotic.jpg
    https://img.pokemondb.net/artwork/castform.jpg
    https://img.pokemondb.net/artwork/kecleon.jpg
    https://img.pokemondb.net/artwork/shuppet.jpg
    https://img.pokemondb.net/artwork/banette.jpg
    https://img.pokemondb.net/artwork/duskull.jpg
    https://img.pokemondb.net/artwork/dusclops.jpg
    https://img.pokemondb.net/artwork/tropius.jpg
    https://img.pokemondb.net/artwork/chimecho.jpg
    https://img.pokemondb.net/artwork/absol.jpg
    https://img.pokemondb.net/artwork/wynaut.jpg
    https://img.pokemondb.net/artwork/snorunt.jpg
    https://img.pokemondb.net/artwork/glalie.jpg
    https://img.pokemondb.net/artwork/spheal.jpg
    https://img.pokemondb.net/artwork/sealeo.jpg
    https://img.pokemondb.net/artwork/walrein.jpg
    https://img.pokemondb.net/artwork/clamperl.jpg
    https://img.pokemondb.net/artwork/huntail.jpg
    https://img.pokemondb.net/artwork/gorebyss.jpg
    https://img.pokemondb.net/artwork/relicanth.jpg
    https://img.pokemondb.net/artwork/luvdisc.jpg
    https://img.pokemondb.net/artwork/bagon.jpg
    https://img.pokemondb.net/artwork/shelgon.jpg
    https://img.pokemondb.net/artwork/salamence.jpg
    https://img.pokemondb.net/artwork/beldum.jpg
    https://img.pokemondb.net/artwork/metang.jpg
    https://img.pokemondb.net/artwork/metagross.jpg
    https://img.pokemondb.net/artwork/regirock.jpg
    https://img.pokemondb.net/artwork/regice.jpg
    https://img.pokemondb.net/artwork/registeel.jpg
    https://img.pokemondb.net/artwork/latias.jpg
    https://img.pokemondb.net/artwork/latios.jpg
    https://img.pokemondb.net/artwork/kyogre.jpg
    https://img.pokemondb.net/artwork/groudon.jpg
    https://img.pokemondb.net/artwork/rayquaza.jpg
    https://img.pokemondb.net/artwork/jirachi.jpg
    https://img.pokemondb.net/artwork/deoxys-normal.jpg
    https://img.pokemondb.net/artwork/turtwig.jpg
    https://img.pokemondb.net/artwork/grotle.jpg
    https://img.pokemondb.net/artwork/torterra.jpg
    https://img.pokemondb.net/artwork/chimchar.jpg
    https://img.pokemondb.net/artwork/monferno.jpg
    https://img.pokemondb.net/artwork/infernape.jpg
    https://img.pokemondb.net/artwork/piplup.jpg
    https://img.pokemondb.net/artwork/prinplup.jpg
    https://img.pokemondb.net/artwork/empoleon.jpg
    https://img.pokemondb.net/artwork/starly.jpg
    https://img.pokemondb.net/artwork/staravia.jpg
    https://img.pokemondb.net/artwork/staraptor.jpg
    https://img.pokemondb.net/artwork/bidoof.jpg
    https://img.pokemondb.net/artwork/bibarel.jpg
    https://img.pokemondb.net/artwork/kricketot.jpg
    https://img.pokemondb.net/artwork/kricketune.jpg
    https://img.pokemondb.net/artwork/shinx.jpg
    https://img.pokemondb.net/artwork/luxio.jpg
    https://img.pokemondb.net/artwork/luxray.jpg
    https://img.pokemondb.net/artwork/budew.jpg
    https://img.pokemondb.net/artwork/roserade.jpg
    https://img.pokemondb.net/artwork/cranidos.jpg
    https://img.pokemondb.net/artwork/rampardos.jpg
    https://img.pokemondb.net/artwork/shieldon.jpg
    https://img.pokemondb.net/artwork/bastiodon.jpg
    https://img.pokemondb.net/artwork/burmy.jpg
    https://img.pokemondb.net/artwork/wormadam-plant.jpg
    https://img.pokemondb.net/artwork/mothim.jpg
    https://img.pokemondb.net/artwork/combee.jpg
    https://img.pokemondb.net/artwork/vespiquen.jpg
    https://img.pokemondb.net/artwork/pachirisu.jpg
    https://img.pokemondb.net/artwork/buizel.jpg
    https://img.pokemondb.net/artwork/floatzel.jpg
    https://img.pokemondb.net/artwork/cherubi.jpg
    https://img.pokemondb.net/artwork/cherrim.jpg
    https://img.pokemondb.net/artwork/shellos.jpg
    https://img.pokemondb.net/artwork/gastrodon.jpg
    https://img.pokemondb.net/artwork/ambipom.jpg
    https://img.pokemondb.net/artwork/drifloon.jpg
    https://img.pokemondb.net/artwork/drifblim.jpg
    https://img.pokemondb.net/artwork/buneary.jpg
    https://img.pokemondb.net/artwork/lopunny.jpg
    https://img.pokemondb.net/artwork/mismagius.jpg
    https://img.pokemondb.net/artwork/honchkrow.jpg
    https://img.pokemondb.net/artwork/glameow.jpg
    https://img.pokemondb.net/artwork/purugly.jpg
    https://img.pokemondb.net/artwork/chingling.jpg
    https://img.pokemondb.net/artwork/stunky.jpg
    https://img.pokemondb.net/artwork/skuntank.jpg
    https://img.pokemondb.net/artwork/bronzor.jpg
    https://img.pokemondb.net/artwork/bronzong.jpg
    https://img.pokemondb.net/artwork/bonsly.jpg
    https://img.pokemondb.net/artwork/mime-jr.jpg
    https://img.pokemondb.net/artwork/happiny.jpg
    https://img.pokemondb.net/artwork/chatot.jpg
    https://img.pokemondb.net/artwork/spiritomb.jpg
    https://img.pokemondb.net/artwork/gible.jpg
    https://img.pokemondb.net/artwork/gabite.jpg
    https://img.pokemondb.net/artwork/garchomp.jpg
    https://img.pokemondb.net/artwork/munchlax.jpg
    https://img.pokemondb.net/artwork/riolu.jpg
    https://img.pokemondb.net/artwork/lucario.jpg
    https://img.pokemondb.net/artwork/hippopotas.jpg
    https://img.pokemondb.net/artwork/hippowdon.jpg
    https://img.pokemondb.net/artwork/skorupi.jpg
    https://img.pokemondb.net/artwork/drapion.jpg
    https://img.pokemondb.net/artwork/croagunk.jpg
    https://img.pokemondb.net/artwork/toxicroak.jpg
    https://img.pokemondb.net/artwork/carnivine.jpg
    https://img.pokemondb.net/artwork/finneon.jpg
    https://img.pokemondb.net/artwork/lumineon.jpg
    https://img.pokemondb.net/artwork/mantyke.jpg
    https://img.pokemondb.net/artwork/snover.jpg
    https://img.pokemondb.net/artwork/abomasnow.jpg
    https://img.pokemondb.net/artwork/weavile.jpg
    https://img.pokemondb.net/artwork/magnezone.jpg
    https://img.pokemondb.net/artwork/lickilicky.jpg
    https://img.pokemondb.net/artwork/rhyperior.jpg
    https://img.pokemondb.net/artwork/tangrowth.jpg
    https://img.pokemondb.net/artwork/electivire.jpg
    https://img.pokemondb.net/artwork/magmortar.jpg
    https://img.pokemondb.net/artwork/togekiss.jpg
    https://img.pokemondb.net/artwork/yanmega.jpg
    https://img.pokemondb.net/artwork/leafeon.jpg
    https://img.pokemondb.net/artwork/glaceon.jpg
    https://img.pokemondb.net/artwork/gliscor.jpg
    https://img.pokemondb.net/artwork/mamoswine.jpg
    https://img.pokemondb.net/artwork/porygon-z.jpg
    https://img.pokemondb.net/artwork/gallade.jpg
    https://img.pokemondb.net/artwork/probopass.jpg
    https://img.pokemondb.net/artwork/dusknoir.jpg
    https://img.pokemondb.net/artwork/froslass.jpg
    https://img.pokemondb.net/artwork/rotom-normal.jpg
    https://img.pokemondb.net/artwork/uxie.jpg
    https://img.pokemondb.net/artwork/mesprit.jpg
    https://img.pokemondb.net/artwork/azelf.jpg
    https://img.pokemondb.net/artwork/dialga.jpg
    https://img.pokemondb.net/artwork/palkia.jpg
    https://img.pokemondb.net/artwork/heatran.jpg
    https://img.pokemondb.net/artwork/regigigas.jpg
    https://img.pokemondb.net/artwork/giratina-altered.jpg
    https://img.pokemondb.net/artwork/cresselia.jpg
    https://img.pokemondb.net/artwork/phione.jpg
    https://img.pokemondb.net/artwork/manaphy.jpg
    https://img.pokemondb.net/artwork/darkrai.jpg
    https://img.pokemondb.net/artwork/shaymin-land.jpg
    https://img.pokemondb.net/artwork/arceus.jpg
    https://img.pokemondb.net/artwork/victini.jpg
    https://img.pokemondb.net/artwork/snivy.jpg
    https://img.pokemondb.net/artwork/servine.jpg
    https://img.pokemondb.net/artwork/serperior.jpg
    https://img.pokemondb.net/artwork/tepig.jpg
    https://img.pokemondb.net/artwork/pignite.jpg
    https://img.pokemondb.net/artwork/emboar.jpg
    https://img.pokemondb.net/artwork/oshawott.jpg
    https://img.pokemondb.net/artwork/dewott.jpg
    https://img.pokemondb.net/artwork/samurott.jpg
    https://img.pokemondb.net/artwork/patrat.jpg
    https://img.pokemondb.net/artwork/watchog.jpg
    https://img.pokemondb.net/artwork/lillipup.jpg
    https://img.pokemondb.net/artwork/herdier.jpg
    https://img.pokemondb.net/artwork/stoutland.jpg
    https://img.pokemondb.net/artwork/purrloin.jpg
    https://img.pokemondb.net/artwork/liepard.jpg
    https://img.pokemondb.net/artwork/pansage.jpg
    https://img.pokemondb.net/artwork/simisage.jpg
    https://img.pokemondb.net/artwork/pansear.jpg
    https://img.pokemondb.net/artwork/simisear.jpg
    https://img.pokemondb.net/artwork/panpour.jpg
    https://img.pokemondb.net/artwork/simipour.jpg
    https://img.pokemondb.net/artwork/munna.jpg
    https://img.pokemondb.net/artwork/musharna.jpg
    https://img.pokemondb.net/artwork/pidove.jpg
    https://img.pokemondb.net/artwork/tranquill.jpg
    https://img.pokemondb.net/artwork/unfezant.jpg
    https://img.pokemondb.net/artwork/blitzle.jpg
    https://img.pokemondb.net/artwork/zebstrika.jpg
    https://img.pokemondb.net/artwork/roggenrola.jpg
    https://img.pokemondb.net/artwork/boldore.jpg
    https://img.pokemondb.net/artwork/gigalith.jpg
    https://img.pokemondb.net/artwork/woobat.jpg
    https://img.pokemondb.net/artwork/swoobat.jpg
    https://img.pokemondb.net/artwork/drilbur.jpg
    https://img.pokemondb.net/artwork/excadrill.jpg
    https://img.pokemondb.net/artwork/audino.jpg
    https://img.pokemondb.net/artwork/timburr.jpg
    https://img.pokemondb.net/artwork/gurdurr.jpg
    https://img.pokemondb.net/artwork/conkeldurr.jpg
    https://img.pokemondb.net/artwork/tympole.jpg
    https://img.pokemondb.net/artwork/palpitoad.jpg
    https://img.pokemondb.net/artwork/seismitoad.jpg
    https://img.pokemondb.net/artwork/throh.jpg
    https://img.pokemondb.net/artwork/sawk.jpg
    https://img.pokemondb.net/artwork/sewaddle.jpg
    https://img.pokemondb.net/artwork/swadloon.jpg
    https://img.pokemondb.net/artwork/leavanny.jpg
    https://img.pokemondb.net/artwork/venipede.jpg
    https://img.pokemondb.net/artwork/whirlipede.jpg
    https://img.pokemondb.net/artwork/scolipede.jpg
    https://img.pokemondb.net/artwork/cottonee.jpg
    https://img.pokemondb.net/artwork/whimsicott.jpg
    https://img.pokemondb.net/artwork/petilil.jpg
    https://img.pokemondb.net/artwork/lilligant.jpg
    https://img.pokemondb.net/artwork/basculin.jpg
    https://img.pokemondb.net/artwork/sandile.jpg
    https://img.pokemondb.net/artwork/krokorok.jpg
    https://img.pokemondb.net/artwork/krookodile.jpg
    https://img.pokemondb.net/artwork/darumaka.jpg
    https://img.pokemondb.net/artwork/darmanitan-standard.jpg
    https://img.pokemondb.net/artwork/maractus.jpg
    https://img.pokemondb.net/artwork/dwebble.jpg
    https://img.pokemondb.net/artwork/crustle.jpg
    https://img.pokemondb.net/artwork/scraggy.jpg
    https://img.pokemondb.net/artwork/scrafty.jpg
    https://img.pokemondb.net/artwork/sigilyph.jpg
    https://img.pokemondb.net/artwork/yamask.jpg
    https://img.pokemondb.net/artwork/cofagrigus.jpg
    https://img.pokemondb.net/artwork/tirtouga.jpg
    https://img.pokemondb.net/artwork/carracosta.jpg
    https://img.pokemondb.net/artwork/archen.jpg
    https://img.pokemondb.net/artwork/archeops.jpg
    https://img.pokemondb.net/artwork/trubbish.jpg
    https://img.pokemondb.net/artwork/garbodor.jpg
    https://img.pokemondb.net/artwork/zorua.jpg
    https://img.pokemondb.net/artwork/zoroark.jpg
    https://img.pokemondb.net/artwork/minccino.jpg
    https://img.pokemondb.net/artwork/cinccino.jpg
    https://img.pokemondb.net/artwork/gothita.jpg
    https://img.pokemondb.net/artwork/gothorita.jpg
    https://img.pokemondb.net/artwork/gothitelle.jpg
    https://img.pokemondb.net/artwork/solosis.jpg
    https://img.pokemondb.net/artwork/duosion.jpg
    https://img.pokemondb.net/artwork/reuniclus.jpg
    https://img.pokemondb.net/artwork/ducklett.jpg
    https://img.pokemondb.net/artwork/swanna.jpg
    https://img.pokemondb.net/artwork/vanillite.jpg
    https://img.pokemondb.net/artwork/vanillish.jpg
    https://img.pokemondb.net/artwork/vanilluxe.jpg
    https://img.pokemondb.net/artwork/deerling.jpg
    https://img.pokemondb.net/artwork/sawsbuck.jpg
    https://img.pokemondb.net/artwork/emolga.jpg
    https://img.pokemondb.net/artwork/karrablast.jpg
    https://img.pokemondb.net/artwork/escavalier.jpg
    https://img.pokemondb.net/artwork/foongus.jpg
    https://img.pokemondb.net/artwork/amoonguss.jpg
    https://img.pokemondb.net/artwork/frillish.jpg
    https://img.pokemondb.net/artwork/jellicent.jpg
    https://img.pokemondb.net/artwork/alomomola.jpg
    https://img.pokemondb.net/artwork/joltik.jpg
    https://img.pokemondb.net/artwork/galvantula.jpg
    https://img.pokemondb.net/artwork/ferroseed.jpg
    https://img.pokemondb.net/artwork/ferrothorn.jpg
    https://img.pokemondb.net/artwork/klink.jpg
    https://img.pokemondb.net/artwork/klang.jpg
    https://img.pokemondb.net/artwork/klinklang.jpg
    https://img.pokemondb.net/artwork/tynamo.jpg
    https://img.pokemondb.net/artwork/eelektrik.jpg
    https://img.pokemondb.net/artwork/eelektross.jpg
    https://img.pokemondb.net/artwork/elgyem.jpg
    https://img.pokemondb.net/artwork/beheeyem.jpg
    https://img.pokemondb.net/artwork/litwick.jpg
    https://img.pokemondb.net/artwork/lampent.jpg
    https://img.pokemondb.net/artwork/chandelure.jpg
    https://img.pokemondb.net/artwork/axew.jpg
    https://img.pokemondb.net/artwork/fraxure.jpg
    https://img.pokemondb.net/artwork/haxorus.jpg
    https://img.pokemondb.net/artwork/cubchoo.jpg
    https://img.pokemondb.net/artwork/beartic.jpg
    https://img.pokemondb.net/artwork/cryogonal.jpg
    https://img.pokemondb.net/artwork/shelmet.jpg
    https://img.pokemondb.net/artwork/accelgor.jpg
    https://img.pokemondb.net/artwork/stunfisk.jpg
    https://img.pokemondb.net/artwork/mienfoo.jpg
    https://img.pokemondb.net/artwork/mienshao.jpg
    https://img.pokemondb.net/artwork/druddigon.jpg
    https://img.pokemondb.net/artwork/golett.jpg
    https://img.pokemondb.net/artwork/golurk.jpg
    https://img.pokemondb.net/artwork/pawniard.jpg
    https://img.pokemondb.net/artwork/bisharp.jpg
    https://img.pokemondb.net/artwork/bouffalant.jpg
    https://img.pokemondb.net/artwork/rufflet.jpg
    https://img.pokemondb.net/artwork/braviary.jpg
    https://img.pokemondb.net/artwork/vullaby.jpg
    https://img.pokemondb.net/artwork/mandibuzz.jpg
    https://img.pokemondb.net/artwork/heatmor.jpg
    https://img.pokemondb.net/artwork/durant.jpg
    https://img.pokemondb.net/artwork/deino.jpg
    https://img.pokemondb.net/artwork/zweilous.jpg
    https://img.pokemondb.net/artwork/hydreigon.jpg
    https://img.pokemondb.net/artwork/larvesta.jpg
    https://img.pokemondb.net/artwork/volcarona.jpg
    https://img.pokemondb.net/artwork/cobalion.jpg
    https://img.pokemondb.net/artwork/terrakion.jpg
    https://img.pokemondb.net/artwork/virizion.jpg
    https://img.pokemondb.net/artwork/tornadus-incarnate.jpg
    https://img.pokemondb.net/artwork/thundurus-incarnate.jpg
    https://img.pokemondb.net/artwork/reshiram.jpg
    https://img.pokemondb.net/artwork/zekrom.jpg
    https://img.pokemondb.net/artwork/landorus-incarnate.jpg
    https://img.pokemondb.net/artwork/kyurem-normal.jpg
    https://img.pokemondb.net/artwork/keldeo-ordinary.jpg
    https://img.pokemondb.net/artwork/meloetta-aria.jpg
    https://img.pokemondb.net/artwork/genesect.jpg
    https://img.pokemondb.net/artwork/chespin.jpg
    https://img.pokemondb.net/artwork/quilladin.jpg
    https://img.pokemondb.net/artwork/chesnaught.jpg
    https://img.pokemondb.net/artwork/fennekin.jpg
    https://img.pokemondb.net/artwork/braixen.jpg
    https://img.pokemondb.net/artwork/delphox.jpg
    https://img.pokemondb.net/artwork/froakie.jpg
    https://img.pokemondb.net/artwork/frogadier.jpg
    https://img.pokemondb.net/artwork/greninja.jpg
    https://img.pokemondb.net/artwork/bunnelby.jpg
    https://img.pokemondb.net/artwork/diggersby.jpg
    https://img.pokemondb.net/artwork/fletchling.jpg
    https://img.pokemondb.net/artwork/fletchinder.jpg
    https://img.pokemondb.net/artwork/talonflame.jpg
    https://img.pokemondb.net/artwork/scatterbug.jpg
    https://img.pokemondb.net/artwork/spewpa.jpg
    https://img.pokemondb.net/artwork/vivillon.jpg
    https://img.pokemondb.net/artwork/litleo.jpg
    https://img.pokemondb.net/artwork/pyroar.jpg
    https://img.pokemondb.net/artwork/flabebe.jpg
    https://img.pokemondb.net/artwork/floette.jpg
    https://img.pokemondb.net/artwork/florges.jpg
    https://img.pokemondb.net/artwork/skiddo.jpg
    https://img.pokemondb.net/artwork/gogoat.jpg
    https://img.pokemondb.net/artwork/pancham.jpg
    https://img.pokemondb.net/artwork/pangoro.jpg
    https://img.pokemondb.net/artwork/furfrou.jpg
    https://img.pokemondb.net/artwork/espurr.jpg
    https://img.pokemondb.net/artwork/meowstic-male.jpg
    https://img.pokemondb.net/artwork/honedge.jpg
    https://img.pokemondb.net/artwork/doublade.jpg
    https://img.pokemondb.net/artwork/aegislash-blade.jpg
    https://img.pokemondb.net/artwork/spritzee.jpg
    https://img.pokemondb.net/artwork/aromatisse.jpg
    https://img.pokemondb.net/artwork/swirlix.jpg
    https://img.pokemondb.net/artwork/slurpuff.jpg
    https://img.pokemondb.net/artwork/inkay.jpg
    https://img.pokemondb.net/artwork/malamar.jpg
    https://img.pokemondb.net/artwork/binacle.jpg
    https://img.pokemondb.net/artwork/barbaracle.jpg
    https://img.pokemondb.net/artwork/skrelp.jpg
    https://img.pokemondb.net/artwork/dragalge.jpg
    https://img.pokemondb.net/artwork/clauncher.jpg
    https://img.pokemondb.net/artwork/clawitzer.jpg
    https://img.pokemondb.net/artwork/helioptile.jpg
    https://img.pokemondb.net/artwork/heliolisk.jpg
    https://img.pokemondb.net/artwork/tyrunt.jpg
    https://img.pokemondb.net/artwork/tyrantrum.jpg
    https://img.pokemondb.net/artwork/amaura.jpg
    https://img.pokemondb.net/artwork/aurorus.jpg
    https://img.pokemondb.net/artwork/sylveon.jpg
    https://img.pokemondb.net/artwork/hawlucha.jpg
    https://img.pokemondb.net/artwork/dedenne.jpg
    https://img.pokemondb.net/artwork/carbink.jpg
    https://img.pokemondb.net/artwork/goomy.jpg
    https://img.pokemondb.net/artwork/sliggoo.jpg
    https://img.pokemondb.net/artwork/goodra.jpg
    https://img.pokemondb.net/artwork/klefki.jpg
    https://img.pokemondb.net/artwork/phantump.jpg
    https://img.pokemondb.net/artwork/trevenant.jpg
    https://img.pokemondb.net/artwork/pumpkaboo-average.jpg
    https://img.pokemondb.net/artwork/gourgeist-average.jpg
    https://img.pokemondb.net/artwork/bergmite.jpg
    https://img.pokemondb.net/artwork/avalugg.jpg
    https://img.pokemondb.net/artwork/noibat.jpg
    https://img.pokemondb.net/artwork/noivern.jpg
    https://img.pokemondb.net/artwork/xerneas.jpg
    https://img.pokemondb.net/artwork/yveltal.jpg
    https://img.pokemondb.net/artwork/zygarde-50.jpg
    https://img.pokemondb.net/artwork/diancie.jpg
    https://img.pokemondb.net/artwork/hoopa-confined.jpg
    https://img.pokemondb.net/artwork/volcanion.jpg
    https://img.pokemondb.net/artwork/rowlet.jpg
    https://img.pokemondb.net/artwork/dartrix.jpg
    https://img.pokemondb.net/artwork/decidueye.jpg
    https://img.pokemondb.net/artwork/litten.jpg
    https://img.pokemondb.net/artwork/torracat.jpg
    https://img.pokemondb.net/artwork/incineroar.jpg
    https://img.pokemondb.net/artwork/popplio.jpg
    https://img.pokemondb.net/artwork/brionne.jpg
    https://img.pokemondb.net/artwork/primarina.jpg
    https://img.pokemondb.net/artwork/pikipek.jpg
    https://img.pokemondb.net/artwork/trumbeak.jpg
    https://img.pokemondb.net/artwork/toucannon.jpg
    https://img.pokemondb.net/artwork/yungoos.jpg
    https://img.pokemondb.net/artwork/gumshoos.jpg
    https://img.pokemondb.net/artwork/grubbin.jpg
    https://img.pokemondb.net/artwork/charjabug.jpg
    https://img.pokemondb.net/artwork/vikavolt.jpg
    https://img.pokemondb.net/artwork/crabrawler.jpg
    https://img.pokemondb.net/artwork/crabominable.jpg
    https://img.pokemondb.net/artwork/oricorio-baile.jpg
    https://img.pokemondb.net/artwork/cutiefly.jpg
    https://img.pokemondb.net/artwork/ribombee.jpg
    https://img.pokemondb.net/artwork/rockruff.jpg
    https://img.pokemondb.net/artwork/lycanroc-midday.jpg
    https://img.pokemondb.net/artwork/wishiwashi-solo.jpg
    https://img.pokemondb.net/artwork/mareanie.jpg
    https://img.pokemondb.net/artwork/toxapex.jpg
    https://img.pokemondb.net/artwork/mudbray.jpg
    https://img.pokemondb.net/artwork/mudsdale.jpg
    https://img.pokemondb.net/artwork/dewpider.jpg
    https://img.pokemondb.net/artwork/araquanid.jpg
    https://img.pokemondb.net/artwork/fomantis.jpg
    https://img.pokemondb.net/artwork/lurantis.jpg
    https://img.pokemondb.net/artwork/morelull.jpg
    https://img.pokemondb.net/artwork/shiinotic.jpg
    https://img.pokemondb.net/artwork/salandit.jpg
    https://img.pokemondb.net/artwork/salazzle.jpg
    https://img.pokemondb.net/artwork/stufful.jpg
    https://img.pokemondb.net/artwork/bewear.jpg
    https://img.pokemondb.net/artwork/bounsweet.jpg
    https://img.pokemondb.net/artwork/steenee.jpg
    https://img.pokemondb.net/artwork/tsareena.jpg
    https://img.pokemondb.net/artwork/comfey.jpg
    https://img.pokemondb.net/artwork/oranguru.jpg
    https://img.pokemondb.net/artwork/passimian.jpg
    https://img.pokemondb.net/artwork/wimpod.jpg
    https://img.pokemondb.net/artwork/golisopod.jpg
    https://img.pokemondb.net/artwork/sandygast.jpg
    https://img.pokemondb.net/artwork/palossand.jpg
    https://img.pokemondb.net/artwork/pyukumuku.jpg
    https://img.pokemondb.net/artwork/type-null.jpg
    https://img.pokemondb.net/artwork/silvally.jpg
    https://img.pokemondb.net/artwork/minior-meteor.jpg
    https://img.pokemondb.net/artwork/komala.jpg
    https://img.pokemondb.net/artwork/turtonator.jpg
    https://img.pokemondb.net/artwork/togedemaru.jpg
    https://img.pokemondb.net/artwork/mimikyu.jpg
    https://img.pokemondb.net/artwork/bruxish.jpg
    https://img.pokemondb.net/artwork/drampa.jpg
    https://img.pokemondb.net/artwork/dhelmise.jpg
    https://img.pokemondb.net/artwork/jangmo-o.jpg
    https://img.pokemondb.net/artwork/hakamo-o.jpg
    https://img.pokemondb.net/artwork/kommo-o.jpg
    https://img.pokemondb.net/artwork/tapu-koko.jpg
    https://img.pokemondb.net/artwork/tapu-lele.jpg
    https://img.pokemondb.net/artwork/tapu-bulu.jpg
    https://img.pokemondb.net/artwork/tapu-fini.jpg
    https://img.pokemondb.net/artwork/cosmog.jpg
    https://img.pokemondb.net/artwork/cosmoem.jpg
    https://img.pokemondb.net/artwork/solgaleo.jpg
    https://img.pokemondb.net/artwork/lunala.jpg
    https://img.pokemondb.net/artwork/nihilego.jpg
    https://img.pokemondb.net/artwork/buzzwole.jpg
    https://img.pokemondb.net/artwork/pheromosa.jpg
    https://img.pokemondb.net/artwork/xurkitree.jpg
    https://img.pokemondb.net/artwork/celesteela.jpg
    https://img.pokemondb.net/artwork/kartana.jpg
    https://img.pokemondb.net/artwork/guzzlord.jpg
    https://img.pokemondb.net/artwork/necrozma.jpg
    https://img.pokemondb.net/artwork/magearna.jpg
    https://img.pokemondb.net/artwork/marshadow.jpg
    https://img.pokemondb.net/artwork/poipole.jpg
    https://img.pokemondb.net/artwork/naganadel.jpg
    https://img.pokemondb.net/artwork/stakataka.jpg
    https://img.pokemondb.net/artwork/blacephalon.jpg
    https://img.pokemondb.net/artwork/tmp/zeraora.jpg



```python
poke_col_list_flat = []
for i in poke_col_list:
    for j in i:
        poke_col_list_flat.append(j)
b, g, r = zip(*poke_col_list_flat) # create new lists that correspond to blue, green, and red channel
```


```python
dataf['b'] = b
dataf['g'] = g
dataf['r'] = r
color = []
for i in range(0,len(dataf)):
    index = dataf.iloc[i]
    col = 'rgb'+str(tuple([index.r, index.g, index.b]))
    color.append(col)
dataf['color'] = color
dataf.pokemon
```




    0        Bulbasaur
    1          Ivysaur
    2         Venusaur
    3       Charmander
    4       Charmeleon
    5        Charizard
    6         Squirtle
    7        Wartortle
    8        Blastoise
    9         Caterpie
    10         Metapod
    11      Butterfree
    12          Weedle
    13          Kakuna
    14        Beedrill
    15          Pidgey
    16       Pidgeotto
    17         Pidgeot
    18         Rattata
    19        Raticate
    20         Spearow
    21          Fearow
    22           Ekans
    23           Arbok
    24         Pikachu
    25          Raichu
    26       Sandshrew
    27       Sandslash
    28        Nidoran♀
    29        Nidorina
              ...     
    777        Mimikyu
    778        Bruxish
    779         Drampa
    780       Dhelmise
    781       Jangmo-o
    782       Hakamo-o
    783        Kommo-o
    784      Tapu Koko
    785      Tapu Lele
    786      Tapu Bulu
    787      Tapu Fini
    788         Cosmog
    789        Cosmoem
    790       Solgaleo
    791         Lunala
    792       Nihilego
    793       Buzzwole
    794      Pheromosa
    795      Xurkitree
    796     Celesteela
    797        Kartana
    798       Guzzlord
    799       Necrozma
    800       Magearna
    801      Marshadow
    802        Poipole
    803      Naganadel
    804      Stakataka
    805    Blacephalon
    806        Zeraora
    Name: pokemon, Length: 807, dtype: object




```python
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

trace1 = go.Scatter3d(
    x=dataf['r'],
    y=dataf['b'],
    z=dataf['g'],
    mode='markers',
    marker=dict(
        color=dataf['color'],
        size=5,
        line=dict(
            color= dataf['color'],
            width=0.5
        ),
        opacity=0.8
    ),
    text = dataf.pokemon
)

data = [trace1]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
        xaxis = dict(title = 'Red'),
        yaxis = dict(title = 'Blue'),
        zaxis = dict(title = 'Green')
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~kennylov/61.embed" height="525px" width="100%"></iframe>



Interesting results... now, keeping the same axis, let's group by type to see if they're clustered.


```python
import random

color_type_dict = {}

for i in np.unique(dataf.type):
    red = random.randint(0,255)
    green = random.randint(0,255)
    blue = random.randint(0,255)
    color_type_dict[i] = 'rgb'+'('+ str(red) +',' + str(green)+',' + str(blue) + ')'

```


```python
color_type = []
for i in range(0,len(dataf)):
    p_type = dataf.iloc[i].type
    color_item = color_type_dict[p_type]
    col = color_item
    color_type.append(col)
    
dataf['color_type'] = color_type


trace2 = go.Scatter3d(
    x=dataf['r'],
    y=dataf['b'],
    z=dataf['g'],
    mode='markers',
    marker=dict(
        color=dataf['color_type'],
        size=5,
        line=dict(
            color= dataf['color_type'],
            width=0.5
        ),
        opacity=0.8
    ),
    text = dataf['type']
)

data2 = [trace2]

layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
        xaxis = dict(title = 'Red'),
        yaxis = dict(title = 'Blue'),
        zaxis = dict(title = 'Green')
    )
)
fig = go.Figure(data=data2, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~kennylov/61.embed" height="525px" width="100%"></iframe>



Just looking at this scatterplot, you can't really see any clusters of pokemon types based on colors... but just out of curiousity I'll try using knn to classify by color.


```python
# split up the datasets
dataf2 = dataf.copy().drop('pokemon', axis=1).as_matrix()
train = dataf2[0:721]
trainX = train[:,[8,9,10]]
trainy = train[:,6]
test = dataf2[721:len(dataf), ]
testX = test[:,[8,9,10]]
testy = test[:,6]
testy2 = test[:,7]

```


```python
from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier(n_neighbors = 15)
knn_clf.fit(trainX, trainy)
print('knn classification rate:', str(sum(knn_clf.predict(testX) == testy)/len(testy)))

```

    svm class rate: 0.13953488372093023
    knn classification rate: 0.09302325581395349


Now trying with stats and color


```python
# split up the datasets
dataf2 = dataf.copy().drop('pokemon', axis=1).as_matrix()
train = dataf2[0:721]
trainX = train[:,[0,1,2,8,9,10]]
trainy = train[:,6]
test = dataf2[721:len(dataf), ]
testX = test[:,[0,1,2,8,9,10]]
testy = test[:,6]
testy2 = test[:,7]
```


```python
from sklearn import neighbors
knn_clf = neighbors.KNeighborsClassifier()
clf = GridSearchCV(estimator = knn_clf, param_grid=dict(n_neighbors = [1,2,3,4,5,6,7,8,9,10]))
clf.fit(trainX, trainy)
clf.best_estimator_
#print('knn classification rate:', str(sum(knn_clf.predict(testX) == testy)/len(testy)))

```

    svm class rate: 0.13953488372093023





    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=9, p=2,
               weights='uniform')


