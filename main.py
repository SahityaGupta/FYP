""" Insérer le script à tester et corriger """

"""Script de calcul du productible du nouveau backend de l'appliweb"""
# Standard imports
# Second-party imports
from typing import List
import pandas as pd
import numpy as np
import pvlib
import pytz
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv

GTI_STC = 1000  # W / m2
T_STC = 25  # °C

pvlib_extraradiation = pvlib.irradiance.get_extra_radiation


class CfgSystem_mini:
    """Classe contenant la configuration du projet.

    Attributes:
        site_name (string): Nom indicatif du projet
        latitude (float): Latitude du projet en ° (positif au nord de l'équateur)
        longitude (float): Longtitude du projet en ° (positif à l'est de Greenwich)
        altitude (float): Altitude du projet en m au desus de la mer
        tz (string): Identifiant de la timezone du projet https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

     Args:
        cfg_system (dict): Dictionaire qui contient les paramètres de la modélisation
        module_database (pd.DataFrame): Base de données de la California
        Energy Commission (CEC) sur les données de modélisation des panneaux
        solaires
    """

    def __init__(self, cfg_system: dict):
        self.site_name = cfg_system.get("projectName", "Centrale de Tarnos")
        self.latitude = float(cfg_system.get("latitude", 43.5408))
        self.longitude = float(cfg_system.get("longitude", -1.461038))
        self.altitude = float(cfg_system.get("altitude", 0))
        self.tz = cfg_system.get("tz", "Europe/Paris")
        self.pv = CfgPv(cfg_system, pd.Series(cfg_system['pv'][
                                                  'pv_module']))
        self.nPv = len(cfg_system['pv']['array']['sub-arrays'])
        # Puissance crête de référence (arrondie au kW)
        self.p_ref = round(float(self.pv.array.sub_arrays.pv_peak))

        self.project_name = None

        # Afin de traiter le cas avec des sous - champs, on retire "pv" du
        # dico et on remplace par "pvFields" qui contiendra la liste des
        # dicos des paramètres des différents sous-champs
        self.pvFields = self.pv.array.sub_arrays


class CfgPv:
    """Classe qui contient les paramètres sur les champs solaires.

    Attributes:
        peak_power_target (float): Puissance crête visée en kWc (la puissance crête réelle dépend du nombre de modules et de leurs caractéristiques)
        albedo (float): Facteur d'albedo de l'environnement des modules PV
        ambient_temp (float): Température ambiante à utiliser par défaut en °C
        wind_speed (float): Vitesse horizontale du vent à utiliser par défaut en m/s
        array (CfgArray): Objet contenant les paramètres généraux sur les panneaux solaires
        transformer (CfgTransformerInverter): Objet contenant les paramétres de modélisation des transformateurs PV
        inverter (CfgTransformerInverter): Objet contenant les paramétres de modélisation des onduleurs PV

    Args:
        cfg_system (dict): Dictionaire qui contient les paramètres de la modélisation
        module_database (pd.DataFrame): Base de données de la California Energy Commission (CEC) sur les données
        de modélisation des panneaux solaires
        """

    def __init__(self, cfg_system: dict, module_database: pd.Series):
        pv_cfg = cfg_system['pv']
        self.peak_power_target = float(pv_cfg.get("peak_power_target", 4000))
        self.array = CfgArray(cfg_system, module_database)


class CfgArray:
    """Classe contenant des paramètres généraux sur les panneaux solaires.

    Attributes:
        double_axis_tracker (bool): Indique si on utilise un tracker à double axe
        single_axis_tracker (bool): Indique si on utilise un tracker à axe simple
        use_tracker (bool): Indique si on utilise un tracker sur les panneaux solaires
        use_bifacial (bool): Indique si on utilise des panneaux bifaciaux
        bifacial_coefficient (float): Coefficient qui modélise le gain de performance des panneaux bifaciaux
        sub_arrays (CfgSubArrays): Objet contenant les paramétres de modélisation des sous-champs PV
        losses (CfgLosses): Objet contenant les paramétres des pertes
    Args:
        cfg_system (dict): Dictionaire qui contient les paramètres de la modélisation
        module_database (pd.DataFrame): Base de données de la California Energy Commission (CEC) sur les données
        de modélisation des panneaux solaires
    """

    def __init__(self, cfg_system: dict, module_database: pd.Series):
        array_dict = cfg_system['pv']['array']
        self.double_axis_tracker = array_dict.get("double_axis_tracker", False)
        self.single_axis_tracker = array_dict.get("single_axis_tracker", False)
        self.use_tracker = self.double_axis_tracker or self.single_axis_tracker
        self.use_bifacial = array_dict.get("use_bifacial", False)
        self.bifacial_coefficient = array_dict.get("bifacial_coefficient",
                                                   1.0588)
        self.sub_arrays = CfgSubArrays(cfg_system, self.use_tracker,
                                       module_database)


class CfgSubArrays:
    """Classe contenant la configuration des différents sous-champs.

    Attributes:
        sub_array_i (CfgSubArray): Objet contenant les paramètres de modélisation d'un seul sous-champ
        pv_peak (float): Puissance crête de la centrale calculée à partir de la modélisation des différents sous-champs

    Args:
        cfg_system (dict): Dictionaire qui contient les paramètres de la modélisation
        module_database (pd.DataFrame): Base de données de la California Energy Commission (CEC) sur les données
        de modélisation des panneaux solaires
        use_tracker (bool): Indique si on utilise un tracker solaire ou non
    """

    def __init__(self, cfg_system: dict, use_tracker: bool,
                 module_database: pd.Series):
        list_sub_array = cfg_system["pv"]['array']['sub-arrays']
        self.add_sub_arrays(list_sub_array, use_tracker, module_database)
        self.pv_peak = self.compute_all_peak_power()

    def add_sub_array(self, dict_sub_array: dict, number: int,
                      use_tracker: bool, module_database: pd.Series):
        """Méthode ajoutant la configuration d'un sous champs à Sub_arrays"""
        attribute_name = 'sub_array_' + str(number)
        setattr(self, attribute_name,
                CfgSubArray(dict_sub_array, use_tracker, module_database))

    def add_sub_arrays(self, list_sub_array: List[dict], use_tracker: bool,
                       module_database: pd.DataFrame):
        """Ajoute tous les sous champs à Sub_arrays"""
        for number, dict_sub_array in enumerate(list_sub_array):
            self.add_sub_array(dict_sub_array, number, use_tracker,
                               module_database)

    def compute_all_peak_power(self):
        """Définition de la méthode permetant de calculer la puissance crête
        de la centrale. Calcule la somme de la puissance crête des
        sous-champs."""
        pv_peak = 0
        for under_camp in list(vars(self)):
            pv_peak += getattr(self, under_camp).p_peak
        return round(pv_peak, 4)


class CfgSubArray:
    """Classe contenant les différents paramètres pour un seul sous-champ.

    Attributes:
        name (string): Nom indicatif du sous-champ PV
        module (string): Référence commerciale des modules PV (selon la base de données de modules lue par Pvlib)
        modules_per_string (int): Nombre de modules en série (par string)
        strings_per_inverter (int): Nombre de strings en parallèle (par onduleur)
        surface_azimuth (float): Azimuth des panneaux en ° (0° plein nord, 90° plein est, etc.)
        surface_tilt (float): Inclinaison des panneaux en ° (0° horizontal, 90° vertical)
        p_peak (float): Puissance crête en kW du sous-champ
    Args:
        dict_sub_array (dict): Dictionaire qui contient les paramètres modélisant le sous-champ
        module_database (pd.DataFrame): Base de données de la California Energy Commission (CEC) sur les données
        de modélisation des panneaux solaires
        use_tracker (bool): Indique si on utilise un tracker solaire ou non
    """

    def __init__(self, dict_sub_array: dict, use_tracker: bool,
                 module_database: pd.DataFrame):
        self.name = dict_sub_array.get("name", "PV 1-1")
        self.module = module_database
        self.surface_azimuth = 0 if use_tracker else dict_sub_array.get(
            "surface_azimuth", 180)
        self.surface_tilt = 0 if use_tracker else dict_sub_array.get(
            "surface_tilt", 0)
        self.modules_per_string = dict_sub_array.get("modules_per_string", 100)
        self.strings_per_inverter = dict_sub_array.get("strings_per_inverter",
                                                       1)
        self.p_peak = self.compute_peak_power()


    def compute_peak_power(self):
        """
        Définition de la méthode permetant de calculer la puissance crête d'un seul sous-champs.
        Args:
            self: CfgSubArray

        Returns:
            Puissance crête d'un seul sous-champs en kW

        """
        module = self.module
        peakPower = (self.modules_per_string * self.strings_per_inverter
                     * module["V_mp_ref"] * module["I_mp_ref"])

        return peakPower / 1000


def pvlib_celltemp(poa_global: pd.Series, temp_air: pd.Series,
                   wind_speed: pd.Series,
                   u_c: float,
                   u_v: float,
                   eta_m: float,
                   alpha_absorption: float) -> pd.Series:
    """Fonction calculant la témpérature des cellules PV drâce à la fonction
    temperature.pvsyst_cell de pvlib.

    Args:
         poa_global (pd.Series): Series conteant l'irradiance effective.
         temp_air (pd.Series): Series conteant la témpérature de l'air.
         wind_speed (pd.Series): Series conteant la vitesse du vent.
         u_c (float): Coefficient combiné de perte de chaleur.
         u_v (float): Facteur combiné de perte de chaleur influencé par le
         vent.
         eta_m (float): Rendement externe du module.
         alpha_absorption (float): Coefficient d'absorption.

    Returns:
         Tcell (pd.Series): Température (en °C) de la cellule Pv.
    """

    # Appel de la fonction PVlib
    Tcell = pvlib.temperature.pvsyst_cell(poa_global, temp_air, wind_speed,
                                          u_c=u_c, u_v=u_v,
                                          eta_m=eta_m,
                                          alpha_absorption=alpha_absorption)
    return Tcell.rename("temp_cell")


def pvlib_sd_param(gti, temp_cell, module, EgRef: float,
                   dEgdT: float) -> pvlib.pvsystem.calcparams_desoto:
    """Fonction qui calcule grâce à Pvlib les paramètres du modèle single
    diode.
    Args:
         gti (pd.Series): Series conteant l'irradiance effective.
         temp_cell (pd.Series): Series conteant la témpérature de la cellule PV.
         module (pd.Series): Series conteant les caractéristique du module PV/
         EgRef (float): Energie de gap à la température de
         référence en unités de eV.
         dEgdT (float): Variation de l'énergie de gap en fonction de la
         témparature en unités de 1/K.

    Returns:
         Param one diode (pvlib.pvsystem.calcparams_desoto): Paramètre
         nécessaire pour le modèle One Diode
    """
    return pvlib.pvsystem.calcparams_desoto(gti, temp_cell,
                                            alpha_sc=module["alpha_sc"],
                                            a_ref=module["a_ref"],
                                            I_L_ref=module["I_L_ref"],
                                            I_o_ref=module["I_o_ref"],
                                            R_sh_ref=module["R_sh_ref"],
                                            R_s=module["R_s"],
                                            EgRef=EgRef, dEgdT=dEgdT)


def pvPlantMaker(module: pd.Series, surface_azimuth: float, surface_tilt:
float,
                 modules_per_string: int, strings_per_inverter: int,
                 latitude: float, longitude: float, tz: str,
                 altitude: float = 0,
                 tracker: bool = False,
                 max_angle: float = 90) -> pvlib.pvsystem.LocalizedPVSystem:
    """Génère un objet de type pvlib.pvsystem.LocalizedPvSystem.

    Args:
         module (pd.Series): Series conteant les caractéristiques du module 
         PV du sous_champ.
         surface_azimuth (float): Orientation des panneaux en dégré selon la 
         convention (Nord : 0° et Est : 90°)
         surface_tilt (float): Inclinaison des panneaux en dégré.
         modules_per_string (int): Nombre de string sur un onduleur.
         strings_per_inverter (int): Nombre de panneau sur une string.
         latitude (float): Latitude du sous_champ.
         longitude (float): Longitude du sous_champ.
         tz (float): Timezone du sous_champ.
         altitude (float): Altitude du sous_champ.
         tracker (float): Booléen indiquant la présence de tracker.
         max_angle (float): ADU

    Returns:
         localizedArray (LocalizedPVSystem): Objet Pvlib représentant un 
         sous_champ.
    """
    # Emplacement
    location = pvlib.location.Location(latitude=latitude, longitude=longitude,
                                       tz=tz, altitude=altitude)

    # Champ PV avec ses caractéristiques
    if tracker:
        array = pvlib.tracking.SingleAxisTracker(module=module,
                                                 surface_azimuth=surface_azimuth,
                                                 surface_tilt=surface_tilt,
                                                 modules_per_string=modules_per_string,
                                                 strings_per_inverter=strings_per_inverter,
                                                 max_angle=max_angle)
        localizedArray = pvlib.tracking.LocalizedSingleAxisTracker(array,
                                                                   location)
    else:
        array = pvlib.pvsystem.PVSystem(module=module,
                                        surface_azimuth=surface_azimuth,
                                        surface_tilt=surface_tilt,
                                        modules_per_string=modules_per_string,
                                        strings_per_inverter=strings_per_inverter)
        localizedArray = pvlib.pvsystem.LocalizedPVSystem(array, location)

    # Renvoie un champ PV localisé
    return localizedArray


def solar_position(data: pd.DataFrame, system_cfg: CfgSystem_mini) -> \
        pd.DataFrame:
    """Lis un DataFrame contenant en indice une series de DatetimeIndex et
    calcule la position du soleil dans le ciel.
    Les grandeurs ajoutées à data sont ; azimuth, elevation, zenith,
    apparent_zenith, apparent_elevation.

    Args:
         data (pd.DataFrame): Dataframe contenant en indice une series de
         DatetimeIndex
         system_cfg (Callable): Classe CfgSystem contenant, la latitude,
         la longitude et l'altitude de la centrale considérée.

    Returns:
         result (pd.DataFrame): Dataframe contenant les series de
         azimuth et elevation
    """
    # Déclaration des variables
    latitude = system_cfg.latitude
    longitude = system_cfg.longitude
    altitude = system_cfg.altitude

    # Extraction des indices de Temps
    time = data.index
    # calcul avec la méthode PVlib du zenith, de l'élévation et de l'azimuth
    solar_angle = pvlib.solarposition.get_solarposition(time, latitude,
                                                        longitude, altitude)
    # calcul du cos zenith
    zenithCosTs = np.cos(np.deg2rad(solar_angle['zenith']))

    # création du Dataframe de résultat
    result = pd.concat([data, solar_angle['azimuth'], solar_angle['elevation'],
                        solar_angle['zenith'], solar_angle[
                            'apparent_zenith'],
                        solar_angle['apparent_elevation']], axis=1)

    # ajout de la colonne cos_zenith, utile pour le calcul du ghi
    result['cos_zenith'] = zenithCosTs

    return result


def split_ghi(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Lis un DataFrame contenant une valeur de ghi et
    applique une méthode de split implémentée dans Pvlib pour obtenir dni et
    dhi à chaque pas de temps.

    Args:
         raw_df (pd.DataFrame): DataFrame contenant une serie de valeur de
         GHI, de zenith et de cos(zenith).

    Returns:
         irradiance_disc (pd.DataFrame): DataFrame contenant les series de
         DNI et DHI.

    """

    ghiTs = raw_df['ghi']
    zenithTs = raw_df['zenith']
    zenithCosTs = raw_df['cos_zenith']

    # Calcul du DNI et du DHI
    dni_df = pvlib.irradiance.disc(ghiTs, zenithTs, ghiTs.index)
    dniTs_disc = dni_df['dni'].fillna(0).clip(lower=0).rename('dni')
    dhiTs_disc = (ghiTs - dniTs_disc * zenithCosTs).rename('dhi')

    irradiance_disc = pd.concat([dniTs_disc, dhiTs_disc], axis=1)

    return irradiance_disc


def compute_solar_irradiance(data: pd.DataFrame,
                             LocPvSystem: pvlib.pvsystem.LocalizedPVSystem) -> pd.DataFrame:
    """Lis un DataFrame contenant les grandeurs disponibles et calcule les
    autres grandeurs nécessaires au calcul du productible.
    Par exemple, si seulemument le ghi est disponible,
    la fonction calcule le dni et le
    dhi avec la méthode de split implémentée dans pvlib.

    Args:
         data (pd.DataFrame): Dataframe contenant des series de grandeurs
         disponibles
         LocPvSystem (pvlib.LocPvSystem): Objet Pvlib qui contient les
         paramètres de localisation d'une centrale PV.


    Returns:
         data (pd.Dataframe): Dataframe contenant les series de
         grandeurs nécessaires au calcul du productible. Les 3 grandeurs
         nécessaires sont ; le GHI, le DNI et le DHI.

    """

    # Gestion de la présence/absence de données
    dni_ok = ('dni' in data)
    dhi_ok = ('dhi' in data)
    ghi_ok = ('ghi' in data)

    # DHI et DNI connus, GHI absent (merge)
    if dni_ok and dhi_ok and not ghi_ok:
        data['ghi'] = data['dhi'] + data['cos_zenith'] * data['dni']

    # GHI et DHI connus
    elif ghi_ok and dhi_ok and not dni_ok:
        raw_dni = (data['ghi'] - data['dhi']) / data['cos_zenith']
        raw_dni[data['cos_zenith'] <= 0] = 0
        data.loc[:, 'dni'] = raw_dni

    # GHI et DNI connus
    elif ghi_ok and dni_ok and not dhi_ok:
        raw_dni = data['dni'] * data['cos_zenith']
        raw_dhi = data['ghi'] - raw_dni.where(raw_dni > 0, 0)
        raw_dhi[raw_dhi < 0] = 0
        data.loc[:, 'dhi'] = raw_dhi

    # GHI connu, DHI et DNI absents (split)
    elif ghi_ok and not (dni_ok and dhi_ok):
        # calcul du DNi et du DHi avec la méthode de split
        irradiance_disc = split_ghi(data)
        # attribution des grandeurs calculées
        data.loc[:, 'dhi'] = irradiance_disc['dhi']
        data.loc[:, 'dni'] = irradiance_disc['dni']

    # GHI, DNI, DHI absents (clearsky)
    elif not (ghi_ok or dni_ok or dhi_ok):
        # Modèle clearsky
        dfClearsky = LocPvSystem.get_clearsky(data.index, model='ineichen',
                                              solar_position=data)
        dfClearsky.index = data.index

        data.loc[:, 'ghi'] = dfClearsky['ghi']
        data.loc[:, 'dni'] = dfClearsky['dni']
        data.loc[:, 'dhi'] = dfClearsky['dhi']
        print(f"/!\ Utilisation prévision/mesure clearsky")


    # Autres cas ADU
    else:
        raise NotImplementedError(
            '/!\ cas de figure non défini pour le calcul du productible PV')

    return data


def compute_solar_angle(solpos: pd.DataFrame, LocPvSystem:
pvlib.pvsystem.LocalizedPVSystem, tracker: bool = False) -> pd.DataFrame:
    """Cette fonction calcule l'angle d'incidence du soleil sur le 
    module. Elle prend en entrée un objet LocPvSystem (objet PVlib) et 
    retourne l'angle d'incidence du soleil sur le module. Si le module 
    possède un tracker, on applique alors la méthode 
    singleaxis disponible dans PVlib. 

    Args:
        solpos (pd.DataFrame): Dataframe contenant des grandeurs indiquant 
        la position du soleil
        tracker (bool): Indique la présence d'un tracker solaire
        LocPvSystem (Callable): Object Pvlib qui contient les 
        paramètres de localisation d'une centrale PV. 

    Returns:
         aoi (pd.Series): Series contenant l'angle d'incidence du 
         soleil sur les modules

    """
    if tracker:
        array_orientation = LocPvSystem.singleaxis(solpos['apparent_zenith'],
                                                   solpos['azimuth'])
        surface_tilt = array_orientation["surface_tilt"]
        surface_azimuth = array_orientation["surface_azimuth"]
    else:
        surface_tilt = LocPvSystem.surface_tilt
        surface_azimuth = LocPvSystem.surface_azimuth

    # Calcul de l'angle d'incidence du soleil sur le module
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solpos['apparent_zenith'],
                               solpos['azimuth'])
    return aoi


def compute_effective_irradiance(solpos: pd.DataFrame, aoi: pd.Series, data:
pd.DataFrame,
                                 dni_extra: pd.Series, surface_tilt: float,
                                 surface_azimuth: float,
                                 pressure: float,
                                 albedo: float,
                                 soiling: float,
                                 iam_b: float) -> pd.DataFrame:
    """Cette fonction calcule l'irradiance effective (aussi appelée GTI) 
    arrivant sur les modules PV. Le GTI quantifie la puissance
     d'un rayonnement électromagnétique frappant par unité de
     surface perpendiculaire à sa direction.

    Args:
        solpos (pd.DataFrame): DataFrame contenant les grandeurs qui
        caractérisent la position du solaire dans le ciel.
        pressure (pd.Series): DataFrame qui contient la pression
        atmosphèrique.
        data (pd.DataFrame): Dataframe contenant les grandeurs nécessaires
        pour le calcul du prodcutible.
        albedo (float): coefficient d'Albedo du panneau solaire.
        airmass (pd.Series):
        dni_extra (pd.Series): Series contenant le DNI extra-terrestre
        surface_tilt (float): Inclinaison du panneau solaire (en degrés).
        surface_azimuth (pd.Series): Orientation du panneau solaire (en 
        degrés).
        aoi (pd.Series): Series qui contient l'angle d'incidence du
        soleil sur le module.
        soiling (float): Coefficient qui carractérise les pertes
        d'encrassement.
        iam_b (float): ADU

    Returns:
         effective_irradiance (pd.Series): Series contenant
         l'irradiance effective.

    """

    # Calcul de la masse d'air relative et absolue traversée par les rayons
    relative_airmass = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'])
    airmass = pvlib.atmosphere.get_absolute_airmass(relative_airmass, pressure)

    # paramètre indiquant qi le gti est présent dans le dataframe contenant
    # les grandeurs initiales
    gti_ok = ('gti' in data)

    # Calcul de l'irradiation totale incidente aux panneaux
    if gti_ok:
        effective_irradiance = data['gti']
    else:
        total_irrad = pvlib.irradiance.get_total_irradiance(surface_tilt,
                                                            surface_azimuth,
                                                            solpos["zenith"],
                                                            solpos["azimuth"],
                                                            data['dni'],
                                                            data['ghi'],
                                                            data['dhi'],
                                                            dni_extra=dni_extra,
                                                            airmass=airmass,
                                                            albedo=albedo,
                                                            model="perez",
                                                            )

        # Calcul des pertes IAM
        iam = pd.DataFrame(index=data.index,
                           data={
                               "direct": pvlib.iam.ashrae(aoi, b=iam_b),
                               "diffuse": 1 - iam_b,
                           })
        iam.fillna(0, inplace=True)

        # Application des pertes IAM
        effective_irradiance = (total_irrad['poa_direct'] * iam["direct"]
                                + total_irrad['poa_diffuse'] * iam["diffuse"])

    # Application des pertes d'encrassement du panneaux
    effective_irradiance *= (1 - soiling)

    # Suppression des valeurs NaN ou < 0
    effective_irradiance = effective_irradiance.fillna(0).clip(lower=0)

    return effective_irradiance


def treat_temp_wind(data: pd.DataFrame,
                    defaultTa: float,
                    defaultWindSpeed: float) -> pd.DataFrame:
    """Cette fonction compense l'absence d'information sur la température et
    la vitesse du vent. Si l'information n'est pas contenue dans le
    Dataframe data, alors une colonne nommée temp (respetivement wind_speed)
    contenant la téméprature (respectivement vitesse du vent) par défaut
    est ajoutée.

    Args:
        data (pd.DataFrame): DataFrame contenant les grandeurs initiales
        pour le calcul du prodcutible.
        defaultTa (float): Temperature par défaut.
        defaultWindSpeed (float): Vitesse du vent par défaut.


    Returns:
         data (pd.DataFrame): DataFrame contenant l'information sur la
         vitesse du vent et sur la température.

    """
    temp_ok = ('temp' in data)
    wind_ok = ('wind_speed' in data)

    if not temp_ok:
        # Suppression des éventuelles NaN au niveau de température
        data['temp'] = defaultTa

    if not wind_ok:
        # Suppression des éventuelles NaN au niveau de la vitesse du vent
        data['wind_speed'] = defaultWindSpeed

    return data


def compute_Ppv_power(LocPvSystem: pvlib.pvsystem.LocalizedPVSystem,
                      gti: pd.Series,
                      temp: pd.Series,
                      wind_speed: pd.Series,
                      nameOut: str,
                      EgRef: float,
                      dEgdT: float,
                      mismatch: float,
                      wiring: float,
                      connection: float,
                      lid: float,
                      u_c: float,
                      u_v: float,
                      alpha_absorption: float,
                      nameplate_rating: float) -> pd.DataFrame:
    """Cette fonction calcule en  appliquant le modèle one diode, à
    partir de l'irradiance effective et des
    informations sur le sous-champ PV, la puissance photovoltaïque DC
    produite .

    Args:
        LocPvSystem (pvlib.pvsystem.LocalizedPVSystem): Modélisation du
        sous-champ au format pvlib.
        gti (pd.Series): Series contenant l'irradiance effective.
        wind_speed (pd.Series): Series contenant la vitesse du vent.
        temp (pd.Series):Series contenant la témpature extérieure.
        nameOut (str): Nom donné à la Series qui contient le productible.
        EgRef (float): Energie de gap à la température de
        référence en unités de eV.
        dEgdT (float): Variation de l'énergie de gap en fonction de la
        témparature en unités de 1/K.
        wiring (float): Perte (en %) ohmique des câbles entre les panneaux.
        mismatch (float): Perte (en %) de mismatch.
        connection (float): Perte (en %) ohmique de la résistance de contact
        entre les panneaux.
        lid (float): Perte (en %) Light Induced Degradation relative à la
        dégradation du modules dans les premières heures d'exposition.
        nameplate_rating (float): Perte d'efficacité des modules
        par rapport à la spécification fabricant.


    Returns:
         Ppv_productible (pd.Series): Puissance DC produite par le sous-champ


    """
    # Calcul de la température des modules PV
    stc_efficiency = (LocPvSystem.module["I_mp_ref"]
                      * LocPvSystem.module["V_mp_ref"]
                      / LocPvSystem.module["A_c"]
                      / GTI_STC)
    Tmodule = pvlib_celltemp(gti, temp, wind_speed,
                             eta_m=(0.75 * stc_efficiency), u_c=u_c,
                             u_v=u_v,
                             alpha_absorption=alpha_absorption)

    # Calcul du modèle une diode
    singlediode_params = pvlib_sd_param(gti, Tmodule,
                                        module=LocPvSystem.module,
                                        EgRef=EgRef,
                                        dEgdT=dEgdT)
    single_diode_out = pvlib.pvsystem.singlediode(*singlediode_params)

    # Calcul du produit du nombre de panneaux PV par sous-champs
    strings_per_inverter = LocPvSystem.strings_per_inverter
    modules_per_string = LocPvSystem.modules_per_string
    number_of_module = strings_per_inverter * modules_per_string

    # Calcul du productible
    Ppv_productible = (
            single_diode_out.p_mp / 1000  # Puissance produite par modules (en kW)
            * number_of_module  # Nombre de module
            * (1 - mismatch)
            * (1 - wiring)
            * (1 - connection)
            * (1 - lid)
            * (1 - nameplate_rating))

    Ppv_productible = Ppv_productible.rename(nameOut)
    Ppv_productible = Ppv_productible.fillna(value=0)

    return Ppv_productible


def PvPower(data: pd.DataFrame, pvObject: pvlib.pvsystem.LocalizedPVSystem,
            cfg_system: CfgSystem_mini,
            tracker: bool, settings: dict) -> pd.DataFrame:
    """Fonction qui calcule le productible PV à partir d'un Dataframe
    d'entrée.

    Args:
        pvObject (pvlib.pvsystem.LocalizedPVSystem): Modélisation du
        sous-champ au format pvlib.
        data (pd.Dataframe): Dataframe contenant les grandeurs nécessaires
        au calcul du productible
        cfg_system (pd.Series): Series contenant la vitesse du vent.
        tracker (bool): Booléan indiquant la présence de tracker
        settings (dict): Dicitonnaire qui contient les valeurs des paramètres



    Returns:
         p_pv (pd.Series): Puissance DC produite par le sous-champ

    """
    # calcul de la position du soleil dans le ciel
    data = solar_position(data, cfg_system)

    # calcul des grandeurs manquantes dans le fichier de données initial
    data = compute_solar_irradiance(data, pvObject)

    # calcul du DNI extraterestre
    dni_extra = pvlib_extraradiation(data.index, method='nrel')
    dni_extra.index = data.index

    # calcul de l'angle d'incidence du soleil sur le panneau
    aoi = compute_solar_angle(data, LocPvSystem=pvObject, tracker=tracker)

    # calcul de l'irradiance effective
    gti = compute_effective_irradiance(solpos=data,
                                       pressure=settings['pressure'],
                                       data=data,
                                       albedo=settings['albedo'],
                                       dni_extra=dni_extra,
                                       surface_azimuth=pvObject.surface_azimuth,
                                       surface_tilt=pvObject.surface_tilt,
                                       soiling=settings['soiling'],
                                       aoi=aoi,
                                       iam_b=settings['iam_b'])

    # gestion de la température et de la vitesse du vent
    data = treat_temp_wind(data, defaultTa=settings['defaultTa'],
                           defaultWindSpeed=settings['defaultWindSpeed'])

    # calcul de la puissance PV produite avec application des pertes
    p_pv = compute_Ppv_power(pvObject, gti, data.temp, data.wind_speed,
                             nameOut=settings['nameOut'],
                             EgRef=settings['EgRef'],
                             dEgdT=settings['dEgdT'],
                             mismatch=settings['mismatch'],
                             wiring=settings['wiring'],
                             connection=settings['connection'],
                             lid=settings['lid'],
                             nameplate_rating=settings['nameplate_rating'],
                             u_c=settings['u_c'],
                             u_v=settings['u_v'],
                             alpha_absorption=settings['alpha_absorption'])

    return p_pv


def get_pvObjectList(cfg_system: CfgSystem_mini) -> (list, list):
    # obtention de la liste des objects étant de la classe CfgSubArray
    """Cette fonction renvoie la liste des sous-champs au format
    CfgSubArray.
    Args:
        cfg_system (CfgSystem_mini): Métadonnées
    Returns:
         pvObjectList (list): liste des sous-champs au format
    CfgSubArray
         adjustmentList (list): Liste des coefficients correcteurs
    """
    list_of_sub_array = [attribute for attribute in
                         cfg_system.pvFields.__dict__.keys() if
                         'sub_array' in attribute]

    pvObjectList = [pvPlantMaker(
        tracker=cfg_system.pv.array.use_tracker,
        latitude=cfg_system.latitude,
        longitude=cfg_system.longitude,
        altitude=cfg_system.altitude,
        module=getattr(cfg_system.pvFields, pvCfg).module,
        tz=cfg_system.tz,
        surface_azimuth=getattr(cfg_system.pvFields, pvCfg).surface_azimuth,
        surface_tilt=getattr(cfg_system.pvFields, pvCfg).surface_tilt,
        modules_per_string=getattr(cfg_system.pvFields,
                                   pvCfg).modules_per_string,
        strings_per_inverter=getattr(cfg_system.pvFields,
                                     pvCfg).strings_per_inverter)
        for pvCfg in list_of_sub_array]

    return pvObjectList


# Calcul du productible théorique de pv_lib
def compute_PVpower_sim(dfData: pd.DataFrame, cfg_system: CfgSystem_mini,
                        settings: dict) -> pd.Series:  # rajouter le type de sortie
    """Cette fonction réalise l'ensemble des étapes qui permettent de
    réaliser le calcul du productible pour le backend de l'appliweb.
    Args:
        dfData (pd.DataFrame): Dataframe contenant les grandeurs pour le calcul
        du productible.
        cfg_system (CfgSystem_mini): Métadonnées
        settings (dict): Dictionnaire de paramètres
    Returns:
         p_pv_dc_mpp (pd.Series): Productible (Ppv_dc_mpp_th) calculé
    """
    pvObjectList = get_pvObjectList(cfg_system)

    # Calcul de la puissance des différents champs PV
    PpvList = [None for _ in pvObjectList]
    for i, locPvSystem in enumerate(pvObjectList):
        counter = i + 1
        # Calcul du profil d'ensoleillement/production PV
        productible = PvPower(data=dfData,
                              pvObject=locPvSystem,
                              cfg_system=cfg_system,
                              tracker=cfg_system.pv.array.use_tracker,
                              settings=settings)

        # Correction pour modules bifaciaux
        productible *= cfg_system.pv.array.bifacial_coefficient

        # Traitement du profil de productible PV
        PpvList[i] = productible.rename(f"Ppv_dc_mpp_th_{counter}")

    PpvList_sumed = sum(PpvList)

    # saturation basse du productible
    PpvList_sumed = PpvList_sumed.clip(lower=0)

    return PpvList_sumed


def compute_ppv_productible(df: pd.DataFrame, metadata: dict, solar: bool
                            ) -> pd.DataFrame:
    """Cette fonction applique deux logiques différentes de calcul du
    productible en fonction des configurations des sites.
    Args:
        df (pd.DataFrame): Dataframe contenant les grandeurs pour le calcul
        du productible.
        metadata (dict): Métadonnées
        solar (bool): Boolén indiquant si on traite le cas solaire ou éolien

    Returns:
         p_pv_dc_mpp (pd.Series): Productible (Ppv_dc_mpp_th) calculé
    """
    value1 = 'P_mpp_ref'
    # Différenciation en fonction du cas solaire et du cas éolien
    if solar:
        value2 = 'P_pv'
    else:
        value2 = 'P_wt'

    # calcul du productible
    df['P_pv_productible_analyzed'] = df[value2]
    # deux cas sont à considérer en fonction de la présence au non de PMS

    # si le PMS n'est pas présent ou que l'écrêtage séquentiel est activé
    if not metadata["with_PMS"] or metadata["use_sequantial"] or not solar:
        # soe au dessus du quel on considèree qu'il y a écrêtage
        soe_high = metadata["soe_high"]
        # quand le soe est suppérieur à soe_high on considère qu'il y a
        # écrétage
        mask = df['soe'] >= soe_high

    else:
        # cas où il y a le PMS
        tolerance_curtail = metadata["tolerance_curtail"]
        Ppv_peak = metadata["Ppv"]
        # définition du critère d'écrêtage : quand le P_pv s'écarte trop du
        # P_pv_setpoint on considère qu'il y a écrétage
        mask = (df['P_pv_setpoint'] - df[
            'P_pv'] < tolerance_curtail * Ppv_peak) & (
                       df['ghi'] > 5)

    df['P_pv_productible_analyzed'][mask] = df[value1]
    # Si la batterie est chargée et que Ppv estimé est inférieur au Ppv
    # mesuré, alors on prend Ppv_estimé = Ppv_mesuré
    df['P_pv_productible_analyzed'] = df['P_pv_productible_analyzed'].clip(lower=df[value2])

    return df


def check_wind_speed_temp(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Cette fonction applique une vérification sur les données de
    température et de vitesse de vent.
    Args:
        df (pd.DataFrame): Dataframe contenant les grandeurs pour le calcul
        du productible.
        settings (dict): Dicitonnaire qui contient les valeurs des paramètres

    Returns:
         p_pv_dc_mpp (pd.Series): Productible (Ppv_dc_mpp_th) calculé

    """
    # définition des grandeurs par défaut
    default_temp = settings['defaultTa']  # °C
    default_wind_speed = settings['defaultWindSpeed']  # m/s
    max_temp = settings['max_temp']
    min_temp = settings['min_temp']
    max_wind_speed = settings['max_wind_speed']
    min_wind_speed = settings['min_wind_speed']

    # verification concernant la tempèrature
    if 'temp' in df.columns:
        df['temp'][df['temp'] < min_temp] = default_temp
        df['temp'][df['temp'] > max_temp] = default_temp

    else:
        df['temp'] = default_temp

    # verification concernant la vitesse du vent
    if 'wind_speed' in df.columns:
        df['wind_speed'][
            df['wind_speed'] < min_wind_speed] = default_wind_speed
        df['wind_speed'][
            df['wind_speed'] > max_wind_speed] = default_wind_speed

    else:
        df['wind_speed'] = default_wind_speed

    return df


def analyze(df: pd.DataFrame, metadata) -> pd.DataFrame:
    """Cette fonction réalise l'ensemble des étapes qui permettent de
    réaliser le calcul du productible pour le backend de l'appliweb.
    Args:
        df (pd.DataFrame): Dataframe contenant les grandeurs pour le calcul
        du productible.
        tools (dict): Métadonnées
    Returns:
         p_pv_dc_mpp (pd.Series): Productible (Ppv_dc_mpp_th) calculé
    """
    settings = metadata['settings']

    # On recale le dataframe sur la timezone du projet.
    df = df.tz_localize(None)
    try:
        df = df.tz_localize(pytz.timezone(metadata["tz"]))
    except pytz.NonExistentTimeError as excep:
        # code qui permet de gérer les instants de passage de l'heure d'hiver à l'heure d'été
        # on supprime l'heure qui pose problème
        freq = getFreq(df)
        index_to_cancel = pd.date_range(start=excep.args[0], end=excep.args[0] + pd.Timedelta(hours=1), freq=freq,
                                        closed='left')
        df = df.drop(index_to_cancel)
        df = df.tz_localize(pytz.timezone(metadata["tz"]))
    # verification concernant les valeurs de température et de vitesse de
    # vent
    df = check_wind_speed_temp(df, settings)

    # extraction de la configuration de la centrale PV
    cfg_system = CfgSystem_mini(metadata)

    # mise en mémoire des anciens indices, en sortie du backend il faut des
    # indice au format UTC alors que pour le calcul du productible il faut
    # des indices timezonés.
    utc_index = df.index
    df_with_new_index = df.tz_convert(cfg_system.tz)

    # calcul de la puissance prédite par le modèle PV
    p_pv_dc_mpp_th = compute_PVpower_sim(df_with_new_index, cfg_system, settings).rename('Ppv_dc_mpp_th')

    # conservation des anciens indices
    p_pv_dc_mpp_th.index = utc_index
    df['Ppv_dc_mpp_th'] = p_pv_dc_mpp_th

    # calcul de la prod PV théorique à partir du ghi mesuré et du modèle
    # PV en
    # tenant compte des dispos
    df['P_mpp'] = df['Ppv_dc_mpp_th'] * (df['dispo_pv'] / 100)

    # On se débarasse des valeurs infinies
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # On se remet en UTC (parce que c'est ce qui avait été défini comme design).
    df = df.tz_localize(None)
    df = df.tz_localize(pytz.UTC)

    return df


def loadPickle(filePath):
    """Renvoie une variable enregistrée en .pkl de manière simplifiée."""
    # Utilisation du "with open()" pour gérer l'ouverture propre des fichiers
    with open(filePath, "rb") as file:
        out = pickle.load(file)

    # Retourne la variable chargée depuis le .pkl
    return out

def load_json(name: str, folder: Path) -> dict:
    """Charge des données depuis un fichier JSON du même répertoire.
    Args:
        name (string): Nom du fichier brut
        folder (Path): Chemin d'accès au dossier contenant le json à ouvrir
    Returns:
        cfg (dict): Fichier json ouvert
    """
    # Chargement des données
    with open(folder / f"{name}.json", "r", encoding="utf-8") as file:
        cfg = json.load(file)
    # Suppression de l'entrée '__ignore__'
    cfg.pop("__ignore__", None)
    return cfg


def getFreq(sDate):
    """Estime le pas de temps d'un timeserie de données."""
    # Extraction des timestamps
    index = sDate.index

    # Utilisation des attributs freq ou inferred_freq si possible, sinon calcul direct
    if index.freq is not None:
        realFreq = pd.to_timedelta(index.freq)
    else:
        steps = index.to_series().diff()
        validSteps = steps.loc[steps.notnull()]
        realFreq = validSteps.mode().iloc[0]

    # Renvoi du pas de temps
    return realFreq

def linear_regresion(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


def logistic_regression(X, y):
    # Convert target variable to binary format
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)


def random_forest_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the random forest regression model
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = regressor.predict(X_test)

    # Calculate the mean squared error of the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

###### Overload Model Parameters
lid = 0.015

u_c = 20

u_v = 0

eta_m = 0.1

iam_b = 0.05

albedo = 0.2

wiring = 0.02

soiling = 0.02

if __name__ == '__main__':
    metadata = load_json('metadata', folder=Path('.'))
    data = pd.DataFrame(pd.read_csv('bou.csv', sep=';',decimal=','))
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp', drop=True)
    data = data.tz_convert('UTC').tz_convert(metadata['tz'])
    # update of the parameter in the json
    metadata['settings']['lid'] = lid
    metadata['settings']['u_c'] = u_c
    metadata['settings']['u_v'] = u_v
    metadata['settings']['eta_m'] = eta_m
    metadata['settings']['iam_b'] = iam_b
    metadata['settings']['albedo'] = albedo
    metadata['settings']['wiring'] = wiring
    metadata['settings']['soiling'] = soiling
    result = analyze(data, metadata)
    result.to_csv('file1.csv')
    with open('test.csv', 'w') as f:
        for key in result:
            f.write(key)
        f.close()
    data = pd.read_csv('bou.csv', sep=';', decimal=',')
    X = data.iloc[:, 3:7].values
    y = data['P_mpp_original'].values
    # z = linear_regresion(X,y)
    # k = logistic_regression(X,y)
    # l = random_forest_regression(X,y)
    a = 1
