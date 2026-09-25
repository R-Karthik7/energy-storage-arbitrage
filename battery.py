class Battery:

    def __init__(
        self,
        capacity_kwh=100,
        initial_energy_kwh=50,
        min_soc=0.10,
        max_soc=0.90,
        max_charge_kw=20,
        max_discharge_kw=20,
        charge_efficiency=0.90,
        discharge_efficiency=0.90
    ):

        self.capacity_kwh = capacity_kwh
        self.energy_kwh = initial_energy_kwh

        self.min_soc = min_soc
        self.max_soc = max_soc

        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

    def get_soc(self):
        return self.energy_kwh / self.capacity_kwh

    def charge(self, power_kw, duration_hours=1):
        """
        Charge the battery for a given duration.
        Returns the actual energy stored in the battery.
        """

        power_kw = min(power_kw, self.max_charge_kw)

        energy_input = power_kw * duration_hours

        energy_stored = energy_input * self.charge_efficiency

        max_energy = self.capacity_kwh * self.max_soc

        available_space = max_energy - self.energy_kwh

        energy_stored = min(energy_stored, available_space)

        self.energy_kwh += energy_stored

        return energy_stored

    def discharge(self, power_kw, duration_hours=1):
        """
        Discharge the battery for a given duration.
        Returns the energy delivered to the grid.
        """

        power_kw = min(power_kw, self.max_discharge_kw)

        energy_requested = power_kw * duration_hours

        min_energy = self.capacity_kwh * self.min_soc

        available_energy = self.energy_kwh - min_energy

        energy_removed = min(energy_requested, available_energy)

        self.energy_kwh -= energy_removed

        energy_delivered = energy_removed * self.discharge_efficiency

        return energy_delivered