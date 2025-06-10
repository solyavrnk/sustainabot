class PackagingSlots:
    """Class to manage the required slots for packaging consultation"""
    
    REQUIRED_SLOTS = {
        "main_product": "What is your business's main product?",
        "product_packaging": "Packaging used for the product and for shipping/delivery",
        "packaging_material": "Material(s) used in your current packaging",
        "packaging_reorder_interval": "How often you reorder packaging (e.g., monthly, quarterly)",
        "packaging_cost_per_order": "Average packaging cost per order (in EUR)",
        "packaging_provider": "Current packaging supplier or provider",
        "packaging_budget": "Current budget allocated for packaging (in EUR)",
        "production_location": "Location where your products are produced",
        "shipping_location": "Main destination(s) where products are shipped",
        "sustainability_goals": "Your business’s sustainability goals related to packaging"
    }
    
    def __init__(self):
        # Initialize all slots as None (unfilled)
        self.slots = {slot: None for slot in self.REQUIRED_SLOTS.keys()}
    
    def is_complete(self) -> bool:
        """Check if all required slots are filled"""
        return all(value is not None for value in self.slots.values())
    
    def get_missing_slots(self) -> list:
        """Get list of unfilled slots"""
        return [slot for slot, value in self.slots.items() if value is None]
    
    def update_slot(self, slot_name: str, value):
        """Update a specific slot with a value"""
        if slot_name in self.slots:
            self.slots[slot_name] = value
    
    def to_dict(self) -> dict:
        """Convert slots to dictionary for JSON serialization"""
        return {
            "slots": self.slots,
            "is_complete": self.is_complete(),
            "missing_slots": self.get_missing_slots()
        }
