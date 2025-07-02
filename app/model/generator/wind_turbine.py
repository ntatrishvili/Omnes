from app.model.generator.generator import Generator, Vector


class Wind(Generator):
    default_vector = Vector.ELECTRICITY
    default_contributes_to = "electric_power_balance"

    def __str__(self):
        production_sum = self["p_pv"].sum() if not self["p_pv"].empty else 0
        return f"Wind '{self.id}' with production sum = {production_sum}"
