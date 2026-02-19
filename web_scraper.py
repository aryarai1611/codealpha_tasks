import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_products():
    url = 'https://books.toscrape.com/'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, 'html.parser')
    
    products = []
    items = soup.find_all('article', class_='product_pod')
    
    for item in items:
        name = item.find('h3').find('a')['title']
        price = item.find('p', class_='price_color').text
        rating = item.find('p', class_='star-rating')['class'][1]
        availability = item.find('p', class_='instock availability').text.strip()
        
        products.append({
            'Product Name': name,
            'Price': price,
            'Rating': rating,
            'Availability': availability
        })
    
    return pd.DataFrame(products)

if __name__ == '__main__':
    df = scrape_products()
    df.to_csv('ecommerce_data.csv', index=False)
    print(f"✓ Scraped {len(df)} products successfully!")
    print(f"\nFirst 5 products:\n{df.head()}")
    print(f"\nData saved to: ecommerce_data.csv")
