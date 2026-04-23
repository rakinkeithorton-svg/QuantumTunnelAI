from django import forms


class TunnelingInputForm(forms.Form):
    barrier_height_ev = forms.FloatField(
        min_value=0.1,
        max_value=20.0,
        initial=10.0,
        label="Barrier height V0 (eV)",
        help_text="Choose the rectangular barrier height in electron-volts.",
    )
    particle_energy_ev = forms.FloatField(
        min_value=0.01,
        max_value=20.0,
        initial=5.0,
        label="Particle energy E (eV)",
        help_text="The current solver models stationary tunneling through a finite rectangular barrier, so require E < V0.",
    )
    barrier_width_nm = forms.FloatField(
        min_value=0.05,
        max_value=5.0,
        initial=1.0,
        label="Barrier width a (nm)",
        help_text="Barrier thickness in nanometers.",
    )

    def clean(self):
        cleaned_data = super().clean()
        barrier_height = cleaned_data.get("barrier_height_ev")
        particle_energy = cleaned_data.get("particle_energy_ev")

        if barrier_height is not None and particle_energy is not None and particle_energy >= barrier_height:
            self.add_error(
                "particle_energy_ev",
                "The present wavefunction and transmission solver is defined for the tunneling regime only, so E must remain strictly below V0.",
            )
        return cleaned_data
