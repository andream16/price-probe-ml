# Dictionaries
items_by_manufacturer = {}
manufacturers_by_item = {}


class Manufacturer:

    # Helpers
    @staticmethod
    def add_item_by_manufacturer_entry(manufacturer: str, item: str):
        if manufacturer not in items_by_manufacturer:
            items_by_manufacturer[manufacturer] = [item]
        else:
            items_by_manufacturer[manufacturer].append(item)

    @staticmethod
    def add_manufacturer_by_item_entry(manufacturer: str, item: str):
        if item not in manufacturers_by_item:
            manufacturers_by_item[item] = [manufacturer]
        else:
            manufacturers_by_item[item].append(manufacturer)
