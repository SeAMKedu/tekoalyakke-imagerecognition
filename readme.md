[seamk_logo]:       /img/Seamk_logo.svg
[epliitto_logo]:    /img/EPLiitto_logo_vaaka_vari.jpg

[training1]:        /img/training_result_hhmm_lock.png
[training2]:        /img/training_result_hhmmcat_lock.png
[training3]:        /img/training_result_time_lock.png

[test1]:            /img/test_result_hhmm_lock.png
[test2]:            /img/test_result_hhmmcat_lock.png
[test3]:            /img/test_result_time_lock.png

[model1]:            /img/model_hhmm_lock.png
[model2]:            /img/model_hhmmcat_lock.png
[model3]:            /img/model_time_lock.png


# Kuvantunnistus

Tässä Tekoäly-AKKE hankkeessa tehdyssä demossa katsomme miten kuvasta voidaan tunnistaa tietoja. Pohjanmateriaalina on webbikameran kuva Frami F:n ruokalan kellosta ja koitamme opettaa neuroverkon tunnistamaan kellonajan kuvan perusteella. Tätä ideaa voisi laajentaa myös muunlaisiin mittareihin yms. 

## Luo ajoympäristö


```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install --upgrade tensorflow pandas numpy matplotlib scikit-learn pydot Pillow tqdm
```

Tensorflown asennus M1 macciin menee tämän tapaisesti, katso ohjeet https://developer.apple.com/metal/tensorflow-plugin/
```
source ~/miniforge3/bin/activate
conda install -c apple tensorflow-deps==2.6.0
python -m venv venv
pip install --upgrade pip wheel
pip install --upgrade tensorflow-macos tensorflow-metal pandas numpy matplotlib scikit-learn pydot Pillow tqdm
```

## Ajo

Seuraava komento ajaa luo, opettaa ja testaa kolme eri mallia. Mallit tallennetaan models-kansioon ja graafit sekä testikuvat löytyvät img-kansiosta. "_lock" päätteiset kuvat on ennalta laskettuja ja niitä käytetään tässä dokumentissa.

```
python image_recognition.py
```

## Materiaali

Mallien opetukseen käytettiin Framin ruokalan webbikamerassa näkyvää kellotaulua. Tämä kamera on aktiivinen vain ruokailuaikoina, noin aamuyhdestätoista iltapäiväyhteen saakka. Tämän vuoksi muita aikoja ei tässä systeemissä ole opetettu. 

Minuutin heitot ajoissa ovat täysin mahdollisia, koska materiaali on merkitty kuvat tallentaneen koneen kellon mukaan (tallennusaika tiedostonnimessä) eikä se täysin ole synkattuna kuvattuun kelloon. Tämän virheen poistaminen vaatisi manuaalisen läpikäynnin ja korjailun.

## Mallit

Todetaksemme, että oikeanlaisen mallin valinnalla on merkitystä, teimme kolme erilaista vaihtoehtoa ja selvitimme niiden opetuksen ja käytön helppoutta. 

Ensimmäisessä mallissa koitamme saada tunneille ja minuuteilla arvon nollan ja yhden välillä ja siitä laskettua varsinaisen kellon ajan. Toisessa mallissa tunnit ja minuutit ovat luokkia ja malli koittaa kertoa oikean luokan (0..23 ja 0..59) mistä voimme muodostaa ajan. Viimeinen malli on yksinkertaistus vielä tästä, tuloksena vain yksi luokka, kellonaika minuutin tarkkuudella.

Opetus kuvista näkee kuinka mallin tarkkuus on kehittynyt. Ohjelmassa on katkaisu, mikäli opetus ei näytä enää kehittyvän eikä se yritä ajaa maksimi epokkimäärää läpi. Tätä lopetuskohtaa tarkastelemalla huomaamme että ensimmäinen malli kesti kauiten opettaa, liki 140 epokkia, kun luokittelu mallit saavuttivat rajansa (ja olivat tarkempia) noin 50 ja 80 epokin kohdilla.

### Tunnit ja minuutit desimaalina

Tämä malli antaa tulokset desimaaliluvun nollan ja yhden väliltä sekä kellonajan tunnille että minuutilla. Kertomalla se 24 tai 60:llä saadaan varsinainen aika. 

Mallissa on eniten heittoja verrattuna muihin, pääasiassa vain tuntien kohdalla, mutta minuutitkin heittävät enemmän kuin luokitelluissa malleissa. Syynä ehkä materiaalin rajoittunut aikaväli. 

`hourminute_wrapper.py` sisältää tämän koodin.

#### Malli 

![model1]

#### Opetus

![training1]

#### Testi

![test1]

### Tunnit ja minuutit luokkina

Toinen malli antaa myös erillisen arvon tunneille ja minuuteille, mutta tässä käytetään luokittelua (10, 11, 12, ... ja 31, 32, 33, ...) desimaalitodennäköisyyden sijaan. 

Tarkkuus on tässä paljon parempi, heitot ovat pääsääntöisesti minuuttien luokkaa. Johtuen kaappaustavasta, osa virheistä on väärin! Malli on tunnistanut ajan oikein, mutta johtuen kellonajan erosta kuvatun kellon ja tallentavan koneen välillä, vertailu kuva on itseasiassa labeloitu väärin.

`hourminute_category_wrapper.py` sisältää tämän koodin.

#### Malli

![model2]

#### Opetus

![training2]

#### Testi

![test2]


### Kellonaika luokkana

Kolmannessa mallissa ajat on luokiteltu tunnit ja minuutit muodossa (11:32, 11:33, 11:34, ...) Tarkkuudessa eroa ei tunti/minuutti kohtaiseen luokitteluun juurikaan ole. 

`hourminute_time_wrapper.py` sisältää tämän koodin.

#### Malli

![model3]

#### Opetus

![training3]

#### Testi

![test3]


## Tekoäly-AKKE hanke

Syksystä 2021 syksyyn 2022 kestävässä hankkeessa selvitetään Etelä-Pohjanmaan alueen yritysten tietoja ja tarpeita tekoälyn käytöstä sekä tuodaan esille tapoja sen käyttämiseen eri tapauksissa, innostaen laajempaa käyttöä tällä uudelle teknologialle alueella. Hanketta on rahoittanut Etelä-Pohjanmaan liitto.

![epliitto_logo]

---

![seamk_logo]
