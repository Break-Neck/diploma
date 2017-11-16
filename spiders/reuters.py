import scrapy
from newsparse.items import NewsItem

import dateutil.parser


class StackSpider(scrapy.Spider):
    name = 'reuters'
    allowed_domains = ['reuters.com']
    base_address = 'http://www.reuters.com/resources/archive/us/{}.html'
    
    def start_requests(self):
        start_year = int(getattr(self, 'start', '2007'))
        end_year = int(getattr(self, 'end', '2017'))
        return [scrapy.Request(self.base_address.format(year), callback=self.parse_year)\
                for year in range(end_year, start_year - 1, -1)]
    
    def parse_year(self, response):
        for day_link in response.xpath('//div[@class="module"]//p/a/@href').extract():
            yield response.follow(response.urljoin(day_link), callback=self.parse_day)

    def parse_day(self, response):
        day = response.xpath('//h1/text()').extract_first().split(', ')[-1] + ' '
        for line_selector in response.xpath('//div[@class="headlineMed"]'):
            if not line_selector.xpath('./a[contains(@href, "article")]'):
                continue
            time = line_selector.xpath('text()').extract_first()[1:]
            item = NewsItem()
            item['date'] = dateutil.parser.parse(day + time)
            news_link = line_selector.xpath('a/@href').extract_first()
            request = response.follow(response.urljoin(news_link), callback=self.parse_news)
            request.meta['item'] = item
            
            yield request

    def parse_news(self, response):
        item = response.meta['item']
        item['source'] = 'Reuters'
        item['news_type'] = response.xpath('//div[re:test(@class, "ArticleHeader_channel_*")]//text()').extract_first()
        item['title'] = response.xpath('//h1/text()').extract_first()
        item['text'] = '\n'.join(response.xpath('//div[re:test(@class, "ArticleBody_body_*")]//text()').extract())

        yield item

