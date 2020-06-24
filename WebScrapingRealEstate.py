# encoding:utf-8
"""
Created on Wed Apr 29 12:45:22 2020

@author: alperen kan
"""
import os
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import sys
import numpy as np
import pandas as pd

import string
tick = time.clock()

# TODO: 'ilceler_list' and 'ilce_adlar' should be scraped from the website

ilceler_list = (["https://www.emlakjet.com/satilik-konut/istanbul-arnavutkoy/",
                "https://www.emlakjet.com/satilik-konut/istanbul-atasehir/",
                "https://www.emlakjet.com/satilik-konut/istanbul-avcilar/",
                "https://www.emlakjet.com/satilik-konut/istanbul-bahcelievler/",
                "https://www.emlakjet.com/satilik-konut/istanbul-bakirkoy/",
                "https://www.emlakjet.com/satilik-konut/istanbul-basaksehir/",
                "https://www.emlakjet.com/satilik-konut/istanbul-besiktas/",
                "https://www.emlakjet.com/satilik-konut/istanbul-beykoz/",
                "https://www.emlakjet.com/satilik-konut/istanbul-beylikduzu/",
                "https://www.emlakjet.com/satilik-konut/istanbul-beyoglu/",
                "https://www.emlakjet.com/satilik-konut/istanbul-buyukcekmece/",
                "https://www.emlakjet.com/satilik-konut/istanbul-catalca/",
                "https://www.emlakjet.com/satilik-konut/istanbul-cekmekoy/",
                "https://www.emlakjet.com/satilik-konut/istanbul-esenler/",
                "https://www.emlakjet.com/satilik-konut/istanbul-esenyurt/",
                "https://www.emlakjet.com/satilik-konut/istanbul-eyup/",
                "https://www.emlakjet.com/satilik-konut/istanbul-fatih/",
                "https://www.emlakjet.com/satilik-konut/istanbul-gaziosmanpasa/",
                "https://www.emlakjet.com/satilik-konut/istanbul-gungoren/",
                "https://www.emlakjet.com/satilik-konut/istanbul-kadikoy/",
                "https://www.emlakjet.com/satilik-konut/istanbul-kagithane/",
                "https://www.emlakjet.com/satilik-konut/istanbul-kartal/",
                "https://www.emlakjet.com/satilik-konut/istanbul-kucukcekmece/",
                "https://www.emlakjet.com/satilik-konut/istanbul-maltepe/",
                "https://www.emlakjet.com/satilik-konut/istanbul-pendik/",
                "https://www.emlakjet.com/satilik-konut/istanbul-sariyer/",
                "https://www.emlakjet.com/satilik-konut/istanbul-sisli/",
                "https://www.emlakjet.com/satilik-konut/istanbul-tuzla/",
                "https://www.emlakjet.com/satilik-konut/istanbul-umraniye/",
                "https://www.emlakjet.com/satilik-konut/istanbul-uskudar/"
                ])
ilce_adlar = (["arnavutkoy",
               "atasehir",
              "avcilar",
              "bahcelievler",
              "bakirkoy",
              "basaksehir",
              "besiktas",
              "beykoz",
              "beylikduzu",
              "beyoglu",
              "buyukcekmece",
              "catalca",
              "cekmekoy",
              "esenler",
              "esenyurt",
              "eyup",
              "fatih",
              "gaziosmanpasa",
              "gungoren",
              "kadikoy",
              "kagithane",
              "kartal",
              "kucukcekmece",
              "maltepe",
              "pendik",
              "sariyer",
              "sisli",
              "tuzla",
              "umraniye",
              "uskudar"
              ])






for x in range(30):
    chromedriver = "Chrome Drivers Path like C:\\...."
    chromedriver = os.path.expanduser(chromedriver)
    print('chromedriver path: {}'.format(chromedriver))
    sys.path.append(chromedriver)
    
    driver = webdriver.Chrome(chromedriver)
    
    #hurriyet_url = "https://www.emlakjet.com/satilik-konut/istanbul-bakirkoy/"
    hurriyet_url = ilceler_list[x]
    driver.get(hurriyet_url)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    listings = soup.find_all("a", class_="w_bXG")
    listings[:5]
    
    listings[0]['href']
    
    house_links = ['https://www.emlakjet.com'+row['href'] for row in listings]
    
    next_button = soup.find('ul', class_= "CteMO").findAll("li")[-1].find("a")['href']
    
    #next_link = ['https://www.emlakjet.com'+row['href'] for row in next_button]
    
    def get_house_links(url, driver, pages=2):
        house_links=[]
        driver.get(url)
        for i in range(pages):
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            listings = soup.find_all("a", class_="w_bXG")
            page_data = ['https://www.emlakjet.com'+row['href'] for row in listings]
            house_links.append(page_data)
            time.sleep(np.random.lognormal(0,1))
            next_button = soup.find('ul', class_= "CteMO").findAll("li")[-1].find("a")['href']
            next_button_link = ['https://www.emlakjet.com'+next_button]
            if i<1:
                driver.get(next_button_link[0])
        
        return house_links
    
    AllHouse = get_house_links(hurriyet_url,driver,pages=2)
    
    def get_html_data(url, driver):
        driver.get(url)
        time.sleep(np.random.lognormal(0,1))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        return soup
    uu = "("
    bb = "-"
    vv = "v"
    def clean2(s):
        if(bb in s):
            ind = s.index('-')
            left = s[0:ind]
            right = s[ind+1:]
            left = int(left)
            right = int(right)
            xx = (left+right)/2
            return xx
        elif(vv in s):
            s = 20
            s = int(s)
            return s
        elif(uu in s):
            ind = s.index(uu)
            s = s[0:ind-1]
            s = int(s)
            return s
        else:
            s = int(s)
            return s
    
    def get_ilce(soup):
        try:
            #ilce = 'Bakirkoy'
            ilce = ilce_adlar[x]
            return ilce
        except:
            return np.nan
    
    def get_price(soup):
        try:
            price = soup.find('div', class_= '_34630').text
            price=price.translate(str.maketrans('', '', string.punctuation))
            price = price[0:-2]
            price = int(price)
            return price
        except:
            return np.nan
    def get_m2(soup):
        try:
            m2= soup.find('div', class_='fFWQu', text = 'Alan').findNext('div').findNext('span').text
            m2 = m2[0:-3]
            m2 = int(m2)
            return m2
        except:
            return np.nan
    
    def get_isitma(soup):
        try:
            isitma= soup.find('div', class_='fFWQu', text = 'Isıtma').findNext('div').text
            return isitma
        except:
            return np.nan
    def get_site(soup):
        try:
            site= soup.find('div', class_='fFWQu', text = 'Site İçerisinde').findNext('div').text
            return site
        except:
            return np.nan
    
    def get_kat(soup):
        try:
            kat= soup.find('div', class_='fFWQu', text = 'Dairenin Katı').findNext('div').text
            return kat
        except:
            return np.nan
    def get_oda(soup):
        try:
            oda= soup.find('div', class_='fFWQu', text = 'Oda Sayısı').findNext('div').text
            return oda
        except:
            return np.nan
    
    def get_yas(soup):
        try:
            yas= soup.find('div', class_='fFWQu', text = 'Bina Yaşı').findNext('div').text
            yas = clean2(yas)
            return yas
        except:
            return np.nan
    
    def get_banyosayisi(soup):
        try:
            banyosayisi = soup.find('div', class_='fFWQu', text = 'Banyo Sayısı').findNext('div').text
            banyosayisi = int(banyosayisi)
            return banyosayisi
        except:
            return np.nan
    def get_esyali(soup):
        try:
            esyali = soup.find('div', class_='fFWQu', text = 'Eşya Durumu').findNext('div').text
            return esyali
        except:
            return np.nan
                
    def flatten_list(house_links):
        house_links_flat=[]
        for sublist in house_links:
            for item in sublist:
                house_links_flat.append(item)
        return house_links_flat
    
    def get_house_data(driver,house_links_flat):
        house_data = []
        for link in house_links_flat:
            soup = get_html_data(link,driver)
            ilce = get_ilce(soup)
            m2 = get_m2(soup)
            isitma = get_isitma(soup)
            site = get_site(soup)
            kat = get_kat(soup)
            oda = get_oda(soup)
            yas = get_yas(soup)
            banyosayisi = get_banyosayisi(soup)
            esyali = get_esyali(soup)
            price = get_price(soup)
            house_data.append([ilce,m2,isitma,site,kat,oda,yas,banyosayisi,esyali,price])
            
        return house_data
    
    house_links_5pages = get_house_links(hurriyet_url,driver,pages=2)
    house_links_flat = flatten_list(house_links_5pages)
    house_data_5pages = get_house_data(driver,house_links_flat)
    
    file_name = "%s_%s.csv" % (str(time.strftime("%Y-%m-%d")), str(time.strftime("%H%M%S")))
    columns = ['ilce','m2','isitma','site','kat','oda','yas','banyosayisi','esyali','price']
    pd.DataFrame(house_data_5pages, columns = columns).to_csv(
        file_name, index = True, encoding = "UTF-8"
    )
    

tock = time.clock()
zaman = tock - tick
print("Time::")
print(zaman, "seconds")





















    



























    





