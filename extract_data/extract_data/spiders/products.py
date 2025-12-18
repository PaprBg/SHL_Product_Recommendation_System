import scrapy
import csv
import os

def saveasCSV(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, mode ="a", newline='', encoding = "utf-8") as csvfile:
        fieldnames = ["product_name", "product_url", "test_type", "product_description", "Remote_testing_available", "Job_levels"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


class ProductsSpider(scrapy.Spider):
    name = "products"
    allowed_domains = ["shl.com"]
    start_urls = ["https://www.shl.com/products/product-catalog/"]

    def parse(self, response):
        product_links = response.css('tr[data-entity-id] td.custom__table-heading__title a::attr(href)').getall()

        self.logger.info(f"Found {len(product_links)} product links")

        for link in product_links:
            full_link = response.urljoin(link)
            print(full_link)   # ✅ THIS WILL PRINT
            yield response.follow(full_link, callback=self.parse_product)

        next_page = response.xpath('(//ul[contains(@class, "pagination")])[last()]//li[contains(@class, "-arrow") and contains(@class, "-next")]/a/@href').get()
        if next_page:
            next_page_url = response.urljoin(next_page)
            yield scrapy.Request(url=next_page_url, callback=self.parse)

    def parse_product(self, response):
        #extracting product_name, product_url, test_type, product_description, Remote_testing_available, Job_levels
        product_name = response.css('h1::text').get(default="").strip()
        product_url = response.url
        product_description = response.xpath("//h4[normalize-space()='Description']/following-sibling::p[1]//text()").get(default="").strip()
        job_levels_text = response.xpath("//h4[contains(text(),'Job')]/following-sibling::p[1]//text()").get(default="")
        job_levels = [lvl.strip() for lvl in job_levels_text.split(",") if lvl.strip()]

        test_type = response.css("p.product-catalogue__small-text:contains('Test Type') span.product-catalogue__key::text").getall()

        remote_testing_available = bool(response.css("p.product-catalogue__small-text:contains('Remote Testing') span.catalogue__circle.-yes"))



        item = {
            "product_name": product_name,
            "product_url": product_url,
            "test_type": test_type,
            "product_description": product_description,
            "Remote_testing_available": remote_testing_available,
            "Job_levels": job_levels
        }

        print(item)
        saveasCSV("products.csv", item)


        
