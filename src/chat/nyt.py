import os
import traceback
import requests
import json

NYT_API_KEY = os.getenv("NYT_API_KEY")
sections = ['arts', 'automobiles', 'books/review', 'business', 'fashion', 'food', 'health', 'home', 'insider', 'magazine', 'movies', 'nyregion', 'obituaries', 'opinion', 'politics', 'realestate', 'science', 'sports', 'sundayreview', 'technology', 'theater', 't-magazine', 'travel', 'upshot', 'us', 'world']

class NYTimes():
    def __init__ (self):
        pass

    def stories(self, section='home'):
        if section not in sections:
            raise KeyError(f'no section {section}')
        try:
            stories = requests.get(f'https://api.nytimes.com/svc/topstories/v2/{section}.json?api-key=TvKkanLr8T42xAUml7MDlUFGXC3G5AxA')
            storiesj = stories.json()
            #print(storiesj.keys())
            #for item in storiesj['results']:
            #    print(f" {item['section']}, {item['item_type']}, {item['title']}")
            return storiesj
        except Exception as e:
            print(str(e))
            traceback.print_exc()
            return {}


    def news_of_the_day(self, sections = ['arts', 'books/review', 'business', 'fashion', 'food', 'health', 'insider', 'science', 'technology', 'us', 'world']):
        """ 
        gets 'home' and sorts stories by sections, picking only sections of interest
        """
        news = {}
        newsj = self.stories('home')
        if type(newsj) != dict or 'results' not in newsj.keys():
            return news
        for item in newsj['results']:
            if item['section'] in sections:
                i_section = item['section']
                if i_section not in news.keys():
                    i_news = []
                    news[i_section] = i_news
                else:
                    i_news = news[i_section]
                i_news.append(item)
        return news

    def headlines(self):
        news = self.news_of_the_day()
        stories = '\n'
        for section in news.keys():
            stories += f'{section.upper()}\n'
            for item in news[section]:
                stories += ' - '+item['title']+'\n'
        stories += '\n'
        return stories, news
    
    def uri(self, title):
        title = title.lower().strip()
        for section in news.keys():
            stories += f'{section.upper()}\n'
            for item in news[section]:
                if item['title'].lower().strip()  == title:
                    return item['uri']
        return None
    
if __name__ == '__main__':
    nyt = NYTimes()
    while True:
        print(f'\n\n{sections}\n')
        stories = nyt.stories(input('section: '))
        print(stories['results'][0]['url'])
        print(stories['results'][0]['uri'])
        print(stories['results'][0]['short_url'])
