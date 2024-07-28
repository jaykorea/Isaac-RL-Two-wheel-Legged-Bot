import xml.etree.ElementTree as ET
import numpy as np
import os
from prettytable import PrettyTable
from .noise_manager import AdditiveNoiseManager


class InertiaManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.Gnoise = AdditiveNoiseManager("gaussian")
        self.table = PrettyTable()
        self.table.field_names = ["Body Name", "Original Mass", "Randomized Mass", "Original Diaginertia", "Randomized Diaginertia", "Balanced Diaginertia"]

    @staticmethod
    def balance_inertia(inertia):
        inertia = sorted(inertia)
        if inertia[0] + inertia[1] < inertia[2]:
            inertia[2] = inertia[0] + inertia[1]
        return inertia

    def randomize_mass(self, inertial, noise_params, body_name):
        if 'mass' in inertial.attrib:
            original_mass = float(inertial.attrib['mass'])
            randomized_mass = np.clip(self.Gnoise.apply(original_mass, mean=noise_params['mass_mean'], std=noise_params['mass_std']), 0.01, 10.0)
            inertial.attrib['mass'] = str(randomized_mass)
            return original_mass, randomized_mass
        return None, None

    def randomize_diaginertia(self, inertial, noise_params, body_name):
        if 'diaginertia' in inertial.attrib:
            original_diaginertia = list(map(float, inertial.attrib['diaginertia'].split()))
            randomized_diaginertia = np.clip(self.Gnoise.apply(original_diaginertia, mean=noise_params['inertia_mean'], std=noise_params['inertia_std']), 0.0, 1.0)
            balanced_diaginertia = self.balance_inertia(randomized_diaginertia)
            inertial.attrib['diaginertia'] = ' '.join(map(str, balanced_diaginertia))
            return original_diaginertia, randomized_diaginertia, balanced_diaginertia
        return None, None, None

    def randomize_inertial(self, specific_bodies_noise):
        """
        specific_bodies_noise: A dictionary where keys are body names and values are dictionaries with 'mass_mean', 'mass_std', 'inertia_mean', 'inertia_std'
        """
        print(f"Original model path: {self.model_path}")

        tree = ET.parse(self.model_path)
        root = tree.getroot()

        for body in root.findall('.//body'):
            body_name = body.attrib.get('name')
            if body_name in specific_bodies_noise:
                noise_params = specific_bodies_noise[body_name]
                for inertial in body.findall('inertial'):
                    original_mass, randomized_mass = self.randomize_mass(inertial, noise_params, body_name)
                    original_diaginertia, randomized_diaginertia, balanced_diaginertia = self.randomize_diaginertia(inertial, noise_params, body_name)
                    self.table.add_row([
                        body_name,
                        f"{original_mass:.6f}" if original_mass is not None else "N/A",
                        f"{randomized_mass:.6f}" if randomized_mass is not None else "N/A",
                        f"{original_diaginertia}" if original_diaginertia is not None else "N/A",
                        f"{randomized_diaginertia}" if randomized_diaginertia is not None else "N/A",
                        f"{balanced_diaginertia}" if balanced_diaginertia is not None else "N/A",
                    ])

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..' ))
        randomized_model_path = os.path.join(root_dir, 'assets', 'randomized_model.xml')

        print(f"Randomized model saved to {randomized_model_path}")
        print(self.table)

        tree.write(randomized_model_path)
        return randomized_model_path