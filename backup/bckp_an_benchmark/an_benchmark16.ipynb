{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "094dd6c2-9921-404e-a4b6-6fe99030dadb",
   "metadata": {},
   "source": [
    "# Benchmarking the code for analysis with artificial data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11ff4402-a8a6-4e6d-8e2f-27cf07609278",
   "metadata": {},
   "source": [
    "created 2025-03-26, by Kananovich\n",
    "updated 2025-04-17 by Kananovich"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "48b497a8-bad5-4a7a-ac08-f6f81e4824c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import scipy.stats as sts\n",
    "from scipy import constants as cnst\n",
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d45df2c5-72d6-4272-97b8-75688a53ebf2",
   "metadata": {},
   "source": [
    "general parameters.\n",
    "camera resolution updated on 2025-04-17 using the data provided by Parth"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d2867fc4-8762-45c0-9990-86dea98f81b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "nP = 1000 #number of particles\n",
    "rho = 1510.0\n",
    "dust_diam = 7.14E-6\n",
    "fps = 1.0E10 # camera framerate in frames per second\n",
    "#fps = 295.0 # camera framerate in frames per second\n",
    "time_step = 1.0 / fps\n",
    "res_meters_per_px = 30.0E-20\n",
    "#res_meters_per_px = 30E-6\n",
    "resol_SI = 1.0 / res_meters_per_px # camera resolution in px/meters\n",
    "dust_mass = 4.0 / 3.0 * np.pi * (dust_diam / 2.0)**3 * rho #mass of the dust particles\n",
    "kin_Tx = 1000.0 #kinetic temperature (in Kelvins) along the x axis\n",
    "kin_Ty = 1000.0 #kinetic temperature (in Kelvins) along the y axis\n",
    "drift_x = 0  # asuming the average x-component of the particles is zero (no drift)\n",
    "left_x_bord = 0\n",
    "right_x_bord = 1751.0 # right border of the field of view in pixels\n",
    "left_x_bord_SI = left_x_bord / resol_SI\n",
    "right_x_bord_SI = right_x_bord / resol_SI #coordinated of the right border\n",
    "    #of the filed of view in meters\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8012b279-6023-42eb-8824-a857ad0d3c29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.3333333333333335e+18"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resol_SI"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "43908d1c-51ce-4f42-b839-6e529c295ec8",
   "metadata": {},
   "source": [
    "## Step 1. Creating an array of artificial velocities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "05f47bb2-55ca-4447-9097-626488869ee1",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_ind = np.arange(0,nP,1,dtype = 'int') # array of particles ID numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4cf003a9-2080-4f8f-8522-c9d0fb314c53",
   "metadata": {},
   "outputs": [],
   "source": [
    "sigma_x = np.sqrt(cnst.k * kin_Tx / dust_mass)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4b9f2d1c-a26b-4bbc-a856-11641e4d9564",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.00021903148058823087"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sigma_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "396004af-6823-48ea-900b-86de750399e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "art_vx_rv = sts.norm(drift_x,sigma_x)\n",
    "arr_sample_vx = art_vx_rv.rvs(nP)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9053c464-7f45-4b49-a895-faa57382e64c",
   "metadata": {},
   "source": [
    "## Step 2. Creating an array of artificial coordinates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "82abe852-8cf3-4d43-9d92-66ff022c70b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "art_x_prev_rv = sts.uniform(left_x_bord_SI, right_x_bord_SI - left_x_bord_SI)\n",
    "arr_sample_prev_x = art_x_prev_rv.rvs(nP)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8b43dc83-c90b-4bc0-aa7f-dc68150660f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_prev_x_inResolChunks = arr_sample_prev_x * resol_SI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c413b3d6-3d6d-4d26-9c60-14ce367d3e02",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_prev_x_inResolChunks_int = arr_prev_x_inResolChunks.astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7ff8d821-9eac-44ff-8327-def50e529d39",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_rough_prev_x = arr_prev_x_inResolChunks_int.astype('float64') / resol_SI\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4665e809-0d3d-468f-9684-3b069f65c7c8",
   "metadata": {},
   "source": [
    "## Step 3. Creating an array of artificial coordinates for the \"next frame\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c77a6b66-03f3-46b6-b7b6-75ff37e7c973",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_next_x = arr_rough_prev_x + arr_sample_vx * time_step"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8ac6cffb-f8fe-4e46-a61d-fdf36f7706d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_next_x_inResolChunks = arr_next_x * resol_SI\n",
    "arr_next_x_inResolChunks_int = arr_next_x_inResolChunks.astype('int64')\n",
    "arr_rough_next_x = arr_next_x_inResolChunks_int.astype('float64') / resol_SI"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f54e2bf0-7961-4ef1-bc66-dcd285d186dc",
   "metadata": {},
   "source": [
    "## Step 4. Calculating the restored velocities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "fa923b4b-444a-4686-8d40-928f27da0da6",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_vx_restored = (arr_rough_next_x - arr_rough_prev_x) / time_step"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cff1b803-1431-46c0-bbd5-a634f0ef8083",
   "metadata": {},
   "source": [
    "## Step 5. Calculating the array of discrepancies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c75fe86f-84a2-4884-accd-14c621f2b4ce",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_discrep_x = np.abs(arr_vx_restored - arr_sample_vx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6506d16c-63ae-49be-9f33-441fe57d3120",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_frac_discrep_x = np.abs(arr_discrep_x / arr_sample_vx) * 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b2b76f60-aee1-498a-a764-472fa6d04449",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.2423192443495058e-06"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_frac_discrep_x.min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d348bcec-5258-48cc-bd25-9ec7d0f9f5e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.854261975210537"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_frac_discrep_x.max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "123a6bce-f218-4727-9fd9-ba4a26b1c25a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0096472693511088"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_frac_discrep_x.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "cafd6b73-50d9-455c-830c-424dacae9c84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>particle</th>\n",
       "      <th>frame</th>\n",
       "      <th>x</th>\n",
       "      <th>vx</th>\n",
       "      <th>real_vx</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000003</td>\n",
       "      <td>-0.000412</td>\n",
       "      <td>-0.000413</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000007</td>\n",
       "      <td>0.000147</td>\n",
       "      <td>0.000147</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000014</td>\n",
       "      <td>0.000592</td>\n",
       "      <td>0.000592</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000002</td>\n",
       "      <td>-0.000205</td>\n",
       "      <td>-0.000205</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0.000017</td>\n",
       "      <td>-0.000083</td>\n",
       "      <td>-0.000082</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1995</th>\n",
       "      <td>995</td>\n",
       "      <td>2</td>\n",
       "      <td>0.000018</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1996</th>\n",
       "      <td>996</td>\n",
       "      <td>2</td>\n",
       "      <td>0.000001</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1997</th>\n",
       "      <td>997</td>\n",
       "      <td>2</td>\n",
       "      <td>0.000013</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1998</th>\n",
       "      <td>998</td>\n",
       "      <td>2</td>\n",
       "      <td>0.000009</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1999</th>\n",
       "      <td>999</td>\n",
       "      <td>2</td>\n",
       "      <td>0.000013</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2000 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      particle  frame         x        vx   real_vx\n",
       "0            0      1  0.000003 -0.000412 -0.000413\n",
       "1            1      1  0.000007  0.000147  0.000147\n",
       "2            2      1  0.000014  0.000592  0.000592\n",
       "3            3      1  0.000002 -0.000205 -0.000205\n",
       "4            4      1  0.000017 -0.000083 -0.000082\n",
       "...        ...    ...       ...       ...       ...\n",
       "1995       995      2  0.000018       NaN       NaN\n",
       "1996       996      2  0.000001       NaN       NaN\n",
       "1997       997      2  0.000013       NaN       NaN\n",
       "1998       998      2  0.000009       NaN       NaN\n",
       "1999       999      2  0.000013       NaN       NaN\n",
       "\n",
       "[2000 rows x 5 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def create_art_vels(tTx, tTy, N_particles, metersPerPx, frps, ro, diam, drift_x, drift_y, left_x, right_x, lefty, right_y):\n",
    "    import numpy as np\n",
    "    import scipy.stats as sts\n",
    "    from scipy import constants as cnst\n",
    "    resol_SI = 1.0 / metersPerPx # camera resolution in px/meters\n",
    "   \n",
    "    dust_mass = 4.0 / 3.0 * np.pi * (diam / 2.0)**3 * ro #mass of the dust particles\n",
    "    sigma_x = np.sqrt(cnst.k * tTx / dust_mass)\n",
    "    sigma_y = np.sqrt(cnst.k * tTy / dust_mass)\n",
    "    left_x_bord_SI = left_x / resol_SI\n",
    "    right_x_bord_SI = right_x / resol_SI #coordinated of the right border\n",
    "    time_step = 1.0 / frps\n",
    "\n",
    "\n",
    "    #Creating the arrays to store data in:\n",
    "    arr_ind = np.arange(0,N_particles,1,dtype = 'int') # array of particles ID numbers\n",
    "    arr_first_frame_no = np.zeros(N_particles, dtype = 'int')\n",
    "    arr_first_frame_no = arr_first_frame_no + int(1)        #array to store the frist frame number\n",
    "    arr_next_frame_no = np.zeros(N_particles, dtype = 'int')\n",
    "    arr_next_frame_no = arr_next_frame_no + int(2)        #array to store the frist frame number\n",
    "\n",
    "    #array to store the \"nonexistent\" data:\n",
    "\n",
    "    arr_nan = np.empty(N_particles)\n",
    "    arr_nan.fill(np.nan)\n",
    "    \n",
    "    \n",
    "\n",
    "    artif_vx_rv = sts.norm(drift_x,sigma_x)\n",
    "    arr_sample_vx = artif_vx_rv.rvs(N_particles)\n",
    "\n",
    "    #Array of artificial coordinates for the \"previous\" frame:\n",
    "    art_x_prev_rv = sts.uniform(left_x_bord_SI, right_x_bord_SI - left_x_bord_SI)\n",
    "    arr_sample_prev_x = art_x_prev_rv.rvs(N_particles)\n",
    "\n",
    "    arr_prev_x_inResolChunks = arr_sample_prev_x * resol_SI\n",
    "    arr_prev_x_inResolChunks_int = arr_prev_x_inResolChunks.astype(int)\n",
    "    arr_rough_prev_x = arr_prev_x_inResolChunks_int.astype('float64') / resol_SI\n",
    "    \n",
    "    ## Step 3. Creating an array of artificial coordinates for the \"next frame\"\n",
    "    arr_next_x = arr_rough_prev_x + arr_sample_vx * time_step\n",
    "    arr_next_x_inResolChunks = arr_next_x * resol_SI\n",
    "    arr_next_x_inResolChunks_int = arr_next_x_inResolChunks.astype('int64')\n",
    "    arr_rough_next_x = arr_next_x_inResolChunks_int.astype('float64') / resol_SI\n",
    "\n",
    "## Step 4: Calculating the \"restored\" velocities:\n",
    "    arr_vx_restored = (arr_rough_next_x - arr_rough_prev_x) / time_step\n",
    "\n",
    "    #saving all the data in the output dataframe:\n",
    "    \n",
    "    #first, create a dataframe storing the data of the first 'video frame':\n",
    "    \n",
    "    dataFirstFrame = {'particle':arr_ind, 'frame':arr_first_frame_no, 'x': arr_rough_prev_x, 'vx':arr_vx_restored, 'real_vx': arr_sample_vx}\n",
    "    first_df = pd.DataFrame(dataFirstFrame)\n",
    "    \n",
    "    #the same for the next video frame:\n",
    "\n",
    "    dataNextFrame = {'particle':arr_ind, 'frame':arr_next_frame_no, 'x': arr_rough_next_x, 'vx':arr_nan, 'real_vx': arr_nan}\n",
    "    next_df = pd.DataFrame(dataNextFrame)\n",
    "    \n",
    "    ret_df = pd.concat([first_df,next_df], ignore_index = True)\n",
    "    return ret_df\n",
    "df = create_art_vels(1200, 1200, 1000, 1.0E-8, 100, 1510, 7.14E-6, 0, 0, 0, 1751, 0, 400)\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "bc5d5a66-9785-4ea4-9aab-1961d3723d69",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_first = df[df['frame'] == 1]\n",
    "arr_vxTheor = np.array(df_first['real_vx'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b7b9987c-c7d9-4e9b-8be1-05b60b7a5ec4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 19 artists>"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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b1+ffPKJTp06dd4au+qrq/lffvmlPvnkk/POOXPM7py3O69fMGzu9VxffPGFcnNzNWfOnKB/jM72Pp944glFR0frhz/8YVB9dcXmXhsbG3XixAkVFRVpypQp2rFjh77zne9o5syZKi8vj6heJekXv/iFRo4cqSFDhig2NlZTpkzR6tWrdfPNNwfVZ48Flvz8fDkcjvNuu3fvliQ5HI5O+xtjuhw/27nPd2efc+dc6NwtLS265557NG7cOL355pt64403NGrUKH3rW99SS0tLRPXa0dGh9vZ2/eIXv9DkyZN144036le/+pXef/997dq1K6J6zcvL04gRI3TXXXd95TEjpdez7d+/X7fffrt+8pOfaNKkSec9z8XW2NX8c8e7c8xQzQmGzb1Kpy/AnT17tjo6OrR69erzdHJ+NvZZXV2tn//859qwYcMl/Rl2p/Zw99rR0SFJuv322/XAAw/ommuuUW5urqZPn97lha/dZWOv0unA8uabb2rbtm2qrq7Wk08+qYULF2rnzp3d6Ooveuy3hBYtWqTZs2efd86wYcP0zjvv6E9/+lOn5/7v//6vU2o7w+PxSDqd/gYPHuwfb2xs9O/j8XjU1tampqamgP9CbWxs9P8itMfjueC5S0tLdfjwYVVWVvo/NigtLdWAAQP061//WrNnz46YXs8cf+TIkf7nv/a1ryk5Odl/5Xqk9Pryyy9r7969ev755yX95S9qcnKyli1bpsceeyxiej2jtrZWt9xyi37wgx/okUceOW9f0unXIioqqtN/oZ1dY1d9dTU/OjpaAwcOPO+cM8fsznm78/oFw+Zez2hvb9esWbNUV1enl19+OejVFdv7fO2119TY2Biw4nnq1Ck9+OCDKi4u1uHDhyOm1+TkZEVHRwf8WytJI0aM0Ouvvx5Un90957n+Wr22tLToX//1X7V161ZNmzZNknT11VerpqZGP/vZzzRx4sRu99ljKyzJyckaPnz4ebe4uDjddNNN8vl8euutt/z7/u///q98Pp//H+VzpaamyuPxqKyszD/W1tam8vJy/z4ZGRmKiYkJmHPs2DHt27fPP6c75z558qT69OkTkBbPPD6TkiOl13HjxkmS3nvvPf+czz77TJ988on/ep1I6fWFF17Q22+/rZqaGtXU1Og//uM/JJ3+R/O+++6LqF6l0ysrEyZM0Pz58/X44493WdO5YmNjlZGREXB+6fS1Tl/V10033dRp/o4dO3T99dcrJibmvHPOHLM75+3O6xcMm3uV/hJW3n//fe3cudP/hhJJfc6bN0/vvPOO/+9kTU2NvF6vHnroIb300ksR1WtsbKxuuOGGgH9rJengwYMX9QPANvfa3t6u9vb2TtcJRkVF+d9Duy2oS3R7yJQpU8zVV19tKisrTWVlpRk9enSnW0Kvuuoqs2XLFv/joqIi43K5zJYtW8zevXvNnXfe2eUtoUOGDDE7d+40e/bsMbfcckuXt4Se79wHDhwwTqfT/Mu//Iupra01+/btM3fddZdxuVzm6NGjEdWrMadv8R01apR54403zN69e8306dPNyJEjL/oWbpt7PduuXbsu+bZmW3vdt2+f+drXvmbmzp1rjh075t8aGxsv2NeZWxbXrVtnamtrTU5OjunXr585fPiwMcaY3NxcM2/ePP/8M7dKPvDAA6a2ttasW7eu062Sb7zxhomKijJFRUXmwIEDpqio6Ctvlfyq83b39QuGrb22t7ebGTNmmCFDhpiampqAP8PW1taI6bMrl3qXkM29btmyxcTExJi1a9ea999/36xatcpERUWZ1157LeJ6zcrKMqNGjTK7du0yH3zwgVm/fr2Ji4szq1evDqpHKwLLp59+aubOnWsSEhJMQkKCmTt3bqc3Dklm/fr1/scdHR3m0UcfNR6PxzidTvPNb37T7N27N2CflpYWs2jRIpOUlGTi4+PN9OnTA25P7u65d+zYYcaNG2dcLpcZMGCAueWWW0xlZWVE9urz+cw999xj+vfvb5KSksx3vvOdTseJlF7PdqmBxeZeH330USOp0zZ06NBu9fZv//ZvZujQoSY2NtZcd911pry83P/c/PnzTVZWVsD8V155xVx77bUmNjbWDBs2zKxZs6bTMf/7v//bXHXVVSYmJsYMHz7cvPDCC0Gdt7uvX7Bs7LWurq7LPz9JZteuXRHTZ1cuNbBc6Jzh7nXdunXm7/7u70xcXJwZM2ZM0N9L0lt6PXbsmLn77ruN1+s1cXFx5qqrrjJPPvmk6ejoCKo/hzH//8N7AAAAS4X9tmYAAIALIbAAAADrEVgAAID1CCwAAMB6BBYAAGA9AgsAALAegQUAAFiPwAIAAKxHYAEAANYjsAAAAOsRWAAAgPUILAAAwHr/D3quaMpfwREVAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "counts_xt, bins_xt = np.histogram(arr_vxTheor, bins = 19)\n",
    "fig_x, ax_x = plt.subplots()\n",
    "arr_bins_centers = 0.5 * (bins_xt[1:] + bins_xt[:-1])\n",
    "ax_x.bar(arr_bins_centers, counts_xt, width = (arr_bins_centers[1] - arr_bins_centers[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3071cba7-4fa0-43d4-b10b-22753b751592",
   "metadata": {},
   "outputs": [],
   "source": [
    "bins_x = arr_bins_centers - 0.5 * (arr_bins_centers[1] - arr_bins_centers[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "ebe41b41-b8eb-47dd-97ec-c1e2f3ce4b7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "bins_x = np.append(bins_x, bins_x[-1] + (bins_x[1] - bins_x[0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "acbd9f1d-e292-49a0-82d5-9bc591840e40",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_vx = np.array(df_first['vx'])\n",
    "counts_xe, bins_xe = np.histogram(arr_vx, bins = bins_x)\n",
    "#counts_xe, bins_xe = np.histogram(arr_vx, bins = 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2cf581b1-742c-4fb7-ab8a-981955c25437",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-7.37670611e-04, -6.57082640e-04, -5.76494668e-04, -4.95906696e-04,\n",
       "       -4.15318725e-04, -3.34730753e-04, -2.54142781e-04, -1.73554810e-04,\n",
       "       -9.29668382e-05, -1.23788666e-05,  6.82091050e-05,  1.48797077e-04,\n",
       "        2.29385048e-04,  3.09973020e-04,  3.90560991e-04,  4.71148963e-04,\n",
       "        5.51736935e-04,  6.32324906e-04,  7.12912878e-04,  7.93500850e-04])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bins_xe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9fd97846-e7a3-4948-9916-d7eb4da01d82",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-7.37670611e-04, -6.57082640e-04, -5.76494668e-04, -4.95906696e-04,\n",
       "       -4.15318725e-04, -3.34730753e-04, -2.54142781e-04, -1.73554810e-04,\n",
       "       -9.29668382e-05, -1.23788666e-05,  6.82091050e-05,  1.48797077e-04,\n",
       "        2.29385048e-04,  3.09973020e-04,  3.90560991e-04,  4.71148963e-04,\n",
       "        5.51736935e-04,  6.32324906e-04,  7.12912878e-04,  7.93500850e-04])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bins_xt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "d8f975d2-d81a-4f68-82ee-55b94e97dff9",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_bins_e_centers = 0.5 * (bins_xe[1:] + bins_xe[:-1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "3144289c-af70-4d78-bb0a-bd6363020893",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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ROE9tv19+2zWSrolsHz88cc6duyY3G53SLvb7ZbP5+vQv6SkRA899FCH9oyMjKjNEQCA85rLFbWhDx8+LFeY43frVUKOkz56McZ0aJOkoqIizZ8/P/i4ra1Nn376qfr37x/sHwgElJGRofr6+uh8TGSJ86VOiVpjFbXGJmqNTZGu1Rijw4cPy+v1hn1stwSW9PR0xcXFdVhNaWho6LDqIklOp1NOpzOkrW/fvp2OnZqaGvP/AUnnT50StcYqao1N1BqbIllruCsr7brlKqHExETl5OSorKwspL2srEyjRo3qjikBAACLddtHQvPnz9fMmTN15ZVX6pprrtHq1at14MAB3Xnnnd01JQAAYKluCyxTp07VJ598osWLF+vQoUPKzs7Wiy++qMGDB5/VeE6nUw8++GCHj45izflSp0StsYpaYxO1xiabau22+7AAAAB0FT9+CAAArEdgAQAA1iOwAAAA6xFYAACA9awILI2NjZo5c6ZcLpdcLpdmzpypf/zjH6c9xhijRYsWyev1Kjk5WWPGjNGuXbtC+jQ3N2vevHlKT09Xnz59NHnyZH344Ydhn7uqqkrXX3+9+vbtq379+ik/P181NTUxWaskrV27VpdeeqmSkpLk8Xg0d+7cmK1Vkj755BMNGjRIDofjjPPribW+/fbbuvXWW5WRkaHk5GQNHTpUv/jFL7pc28qVK5WZmamkpCTl5ORo69atp+1fXl6unJwcJSUl6eKLL9bjjz/eoc/zzz+vYcOGyel0atiwYdq4cWPY5+3K8xcuG2ttbW3VwoULNWLECPXp00der1ff+973dPDgwZiq82SzZ8+Ww+HQihUrwq7vbM8pfbm17t69W5MnT5bL5VJKSoquvvpqHThwIOZqPXLkiObOnatBgwYFX4NO/OHjLjMWmDhxosnOzjYVFRWmoqLCZGdnm0mTJp32mCVLlpiUlBTz/PPPmx07dpipU6eagQMHmkAgEOxz5513mq9+9aumrKzMbN++3YwdO9aMHDnSfPHFF10+dyAQMP369TO33367effdd83OnTvNd77zHTNgwADT0tISU7UaY8wjjzxivF6v+c1vfmP27t1rdu7caTZt2hR2nT2h1nY33XST+eY3v2kkmcbGxpirdc2aNWbevHnm1VdfNX/961/N008/bZKTk81jjz12xrrWr19vEhISzBNPPGFqa2vN3Xffbfr06WP279/faf8PPvjA9O7d29x9992mtrbWPPHEEyYhIcE899xzwT4VFRUmLi7OFBcXm927d5vi4mITHx9v3nzzzbDO25XnLxy21vqPf/zDjBs3zjz77LPm3XffNZWVlSY3N9fk5OTEVJ0n2rhxoxk5cqTxer3m0UcfPas6ba917969Ji0tzdx3331m+/bt5q9//av54x//aP72t7/FXK3/8R//Yf7lX/7FbNmyxdTV1Zlf/epXJi4uzrzwwgth1djtgaW2ttZICnkCKisrjSTz7rvvdnpMW1ub8Xg8ZsmSJcG2zz//3LhcLvP4448bY47/kSckJJj169cH+3z00UemV69e5n//93+7fO6qqiojyRw4cCDY55133jGSzN69e2Oq1k8//dQkJyebl19+Oay6emKt7VauXGny8vLMn//857MOLD2l1hPNmTPHjB079oy1ff3rXzd33nlnSNuQIUNMYWFhp/0XLFhghgwZEtI2e/Zsc/XVVwcfT5kyxUycODGkz4QJE8y0adO6fN6uPH/hsrXWzrz11ltG0infjE7H9jo//PBD89WvftXs3LnTDB48+JwCi821Tp061dx2223hFXQaNtc6fPhws3jx4pA+V1xxhXnggQe6UNk/dftHQpWVlXK5XMrNzQ22XX311XK5XKqoqOj0mLq6Ovl8PuXn5wfbnE6n8vLygsdUV1ertbU1pI/X61V2dnawT1fOfckllyg9PV1r1qxRS0uLmpqatGbNGg0fPjzsm9zZXmtZWZna2tr00UcfaejQoRo0aJCmTJmi+vr6sOrsCbVKUm1trRYvXqxf//rX6tXr7P8UekKtJ/P7/UpLSzttXS0tLaqurg45vyTl5+efcuzKysoO/SdMmKBt27aptbX1tH3ax+zKebvy/IXD5lo74/f75XA4Tvmbaqdie51tbW2aOXOm7rvvPg0fPjys2k5mc61tbW3605/+pH/913/VhAkTNGDAAOXm5uqFF16IuVol6dprr9WmTZv00UcfyRijLVu2aM+ePZowYUJYdXZ7YPH5fBowYECH9gEDBnT4ccQTj5HU4YcS3W53cJ/P51NiYqL69et32j5nOndKSopeffVVPfPMM0pOTtZXvvIVvfTSS3rxxRcVHx/ejYJtr/WDDz5QW1ubiouLtWLFCj333HP69NNPNX78eLW0tMRUrc3Nzbr11lv1s5/9TBdeeGFYtXU2b5trPVllZaV+97vfafbs2aet6+OPP9axY8dOO8fO6uqs/xdffKGPP/74tH3ax+zKebvy/IXD5lpP9vnnn6uwsFDTp08P+8fobK9z6dKlio+P1w9/+MOw6uqMzbU2NDToyJEjWrJkiSZOnKjNmzfr29/+tm655RaVl5fHVK2S9Mtf/lLDhg3ToEGDlJiYqIkTJ2rlypW69tprw6ozaoFl0aJFcjgcp922bdsmSXI4HB2ON8Z02n6ik/d35ZiT+5zp3E1NTbrjjjs0evRovfnmm3rjjTc0fPhwfetb31JTU1NM1drW1qbW1lb98pe/1IQJE3T11Vfrt7/9rd5//31t2bIlpmotKirS0KFDddttt51yzFip9US7du3STTfdpJ/+9KcaP378ac9ztnPsrP/J7V0ZM1J9wmFzrdLxL+BOmzZNbW1tWrly5WkqOT0b66yurtYvfvELrV279pz+Dbsy9+6uta2tTZJ000036Z577tFll12mwsJCTZo0qdMvvnaVjbVKxwPLm2++qU2bNqm6ulqPPPKI5syZo5dffrkLVf1T1H5LaO7cuZo2bdpp+1x00UV655139Le//a3Dvr///e8dUls7j8cj6Xj6GzhwYLC9oaEheIzH41FLS4saGxtD/g+1oaEh+IvQHo/njOdet26d9u3bp8rKyuDHBuvWrVO/fv30hz/8QdOmTYuZWtvHHzZsWHD/BRdcoPT09OA312Ol1ldeeUU7duzQc889J+mff6jp6em6//779dBDD8VMre1qa2t13XXX6Qc/+IEeeOCB09YlHX8u4uLiOvwf2olz7KyuzvrHx8erf//+p+3TPmZXztuV5y8cNtfarrW1VVOmTFFdXZ1eeeWVsFdXbK9z69atamhoCFnxPHbsmO69916tWLFC+/bti5la09PTFR8fH/JaK0lDhw7V66+/HladXT3nyb6sWpuamvTjH/9YGzdu1A033CBJuvTSS1VTU6Of//znGjduXJfrjNoKS3p6uoYMGXLaLSkpSddcc438fr/eeuut4LH/93//J7/fH3xRPllmZqY8Ho/KysqCbS0tLSovLw8ek5OTo4SEhJA+hw4d0s6dO4N9unLuo0ePqlevXiFpsf1xe0qOlVpHjx4tSXrvvfeCfT799FN9/PHHwe/rxEqtzz//vN5++23V1NSopqZG//3f/y3p+IvmXXfdFVO1SsdXVsaOHatZs2bp4Ycf7nROJ0tMTFROTk7I+aXj33U6VV3XXHNNh/6bN2/WlVdeqYSEhNP2aR+zK+ftyvMXDptrlf4ZVt5//329/PLLwTeUWKpz5syZeuedd4J/kzU1NfJ6vbrvvvv00ksvxVStiYmJuuqqq0JeayVpz549Z/UDwDbX2traqtbW1g7fE4yLiwu+h3ZZWF/RjZKJEyeaSy+91FRWVprKykozYsSIDpeEXnLJJWbDhg3Bx0uWLDEul8ts2LDB7Nixw9x6662dXhI6aNAg8/LLL5vt27eb6667rtNLQk937t27dxun02n+8z//09TW1pqdO3ea2267zbhcLnPw4MGYqtWY45f4Dh8+3Lzxxhtmx44dZtKkSWbYsGFnfQm3zbWeaMuWLed8WbOtte7cudNccMEFZsaMGebQoUPBraGh4Yx1tV+yuGbNGlNbW2sKCgpMnz59zL59+4wxxhQWFpqZM2cG+7dfKnnPPfeY2tpas2bNmg6XSr7xxhsmLi7OLFmyxOzevdssWbLklJdKnuq8XX3+wmFrra2trWby5Mlm0KBBpqamJuTfsLm5OWbq7My5XiVkc60bNmwwCQkJZvXq1eb99983jz32mImLizNbt26NuVrz8vLM8OHDzZYtW8wHH3xgnnzySZOUlGRWrlwZVo1WBJZPPvnEzJgxw6SkpJiUlBQzY8aMDm8cksyTTz4ZfNzW1mYefPBB4/F4jNPpNN/4xjfMjh07Qo5pamoyc+fONWlpaSY5OdlMmjQp5PLkrp578+bNZvTo0cblcpl+/fqZ6667zlRWVsZkrX6/39xxxx2mb9++Ji0tzXz729/uME6s1Hqicw0sNtf64IMPGkkdtsGDB3eptv/6r/8ygwcPNomJieaKK64w5eXlwX2zZs0yeXl5If1fffVVc/nll5vExERz0UUXmVWrVnUY8/e//7255JJLTEJCghkyZIh5/vnnwzpvV5+/cNlYa11dXaf/fpLMli1bYqbOzpxrYDnTObu71jVr1pivfe1rJikpyYwcOTLs+5L0lFoPHTpkbr/9duP1ek1SUpK55JJLzCOPPGLa2trCqs9hzP//8B4AAMBS3X5ZMwAAwJkQWAAAgPUILAAAwHoEFgAAYD0CCwAAsB6BBQAAWI/AAgAArEdgAQAA1iOwAAAA6xFYAACA9QgsAADAegQWAABgvf8HyMemiw1AcQgAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ax_x.bar(arr_bins_e_centers, counts_xe, width = (arr_bins_centers[1] - arr_bins_centers[0]), color = 'red')\n",
    "fig_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "320a93fb-133e-4ce1-b09f-e1ee664016ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<BarContainer object of 19 artists>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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yeq68sorO/XfvHmzLrvsMiUlJZ2yT8c+u3Pc7rx/4bC5VulfYeXdd9/Vli1bgieUeKpz5syZeuutt4J/k7W1tfJ6vZo/f75eeOGFuKo1OTlZl19+echnrSTt2bPnjH4A2OZa29ra1NbW1uk6wYSEhOA5tNvCukQ3SiZOnGguvvhiU1VVZaqqqsyIESM63RJ60UUXmQ0bNgRfL1261LhcLrNhwwazY8cOc9NNN3V5S+igQYPMli1bzPbt283VV1/d5S2hpzr27t27jdPpND/84Q9NXV2d2blzp7n55puNy+UyBw4ciKtajTl2i+/w4cPNq6++anbs2GEmTZpkhg0bdsa3cNtc6/G2bt161rc121rrzp07zXnnnWdmzJhhDh48GFwaGxtPW1fHLYtr1641dXV1pqioyPTp08e8//77xhhjFi1aZGbOnBns33Gr5J133mnq6urM2rVrO90q+eqrr5qEhASzdOlSs3v3brN06dKT3ip5suN29/0Lh621trW1mcmTJ5tBgwaZ2trakH/DlpaWuKmzK2d7l5DNtW7YsMEkJSWZNWvWmHfffdc8/PDDJiEhwWzbti3uai0oKDDDhw83W7duNe+995557LHHTEpKilm1alVYNVoRWD7++GMzY8YMk5aWZtLS0syMGTM6nTgkmcceeyz4ur293dx3333G4/EYp9NpvvGNb5gdO3aEbNPc3Gzmzp1rMjIyTGpqqpk0aVLI7cndPfbmzZvN6NGjjcvlMv369TNXX321qaqqista/X6/ufXWW03fvn1NRkaG+fa3v91pP/FS6/HONrDYXOt9991nJHVaBg8e3K3afvWrX5nBgweb5ORkc+mll5qKiorgulmzZpmCgoKQ/i+99JK55JJLTHJysrngggvM6tWrO+3zj3/8o7noootMUlKSGTJkiHn22WfDOm53379w2VhrfX19l/9+kszWrVvjps6unG1gOd0xY13r2rVrzVe+8hWTkpJiRo4cGfZzSXpKrQcPHjS33HKL8Xq9JiUlxVx00UXmoYceMu3t7WHV5zDm/395DwAAYKmY39YMAABwOgQWAABgPQILAACwHoEFAABYj8ACAACsR2ABAADWI7AAAADrEVgAAID1CCwAAMB6BBYAAGA9AgsAALAegQUAAFjv/wHQdKyWTTYl0gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig_x_e, ax_x_e = plt.subplots()\n",
    "ax_x_e.bar(arr_bins_e_centers, counts_xe, width = (arr_bins_centers[1] - arr_bins_centers[0]), color = 'red')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "313618ff-ffe8-431f-8732-3fdf326b198c",
   "metadata": {},
   "outputs": [],
   "source": [
    "std = np.std(arr_vx)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e28559cc-c379-4e0f-8f28-e7e37ba0bc9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.00023667353513014502"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "std"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "b1dbb081-061a-4ade-9bfb-0d8836dae398",
   "metadata": {},
   "outputs": [],
   "source": [
    "tT = std**2 * dust_mass / cnst.k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "495a6d0c-d502-46e7-b8d0-4ea67efd4fdf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1167.5791145310952"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "465cabe1-d01d-4ce5-bddf-43bd91bb91c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "arr_resol = np.arange(1E-6,30E-6,1E-6)\n",
    "arr_framerate = np.arange(30.0, 300.0, 30.0)\n",
    "arr_discrep_average = np.zeros((len(arr_resol), len(arr_framerate)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "f5505eda-f180-48f9-a269-4465e94785c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "for k in range(0, 5):\n",
    "    arr_discrep = np.zeros((len(arr_resol), len(arr_framerate)))\n",
    "    for j in range(0, len(arr_framerate)):\n",
    "        frmrt_j = arr_framerate[j]\n",
    "        for i in range(0, len(arr_resol)):\n",
    "            res_i = arr_resol[i]\n",
    "            df_i = create_art_vels(kin_Tx, kin_Ty, 1000, res_i, frmrt_j, 1510, 7.14E-6, 0, 0, 0, 1751, 0, 400)\n",
    "            df_i_ff = df_i[df_i['frame'] == 1]\n",
    "            std_i = np.std(df_i['vx'])\n",
    "            tT = std_i**2 * dust_mass / cnst.k\n",
    "            arr_discrep[i, j] = np.abs((kin_Tx - std_i**2 * dust_mass / cnst.k)) / kin_Tx\n",
    "    arr_discrep_average = arr_discrep_average + arr_discrep / 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "2e84c063-4205-40c7-9028-9e2468e550be",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(0, len(arr_resol)):\n",
    "    res_i = arr_resol[i]\n",
    "    df_i = create_art_vels(kin_Tx, kin_Ty, 1000, res_i, 100, 1510, 7.14E-6, 0, 0, 0, 1751, 0, 400)\n",
    "    df_i_ff = df_i[df_i['frame'] == 1]\n",
    "    std_i = np.std(df_i['vx'])\n",
    "    tT = std_i**2 * dust_mass / cnst.k\n",
    "    arr_discrep[i] = np.abs((kin_Tx - std_i**2 * dust_mass / cnst.k)) / kin_Tx\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "3b3c5eee-22a7-4a02-83a1-c79ec480ec83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'discrepancy (%)')"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots()\n",
    "ax.plot(arr_resol * 1000000, arr_discrep)\n",
    "#ax.set_xscale('log')\n",
    "ax.set_xlabel(r\"resolution ($\\mu$m/px)\")\n",
    "ax.set_ylabel(\"discrepancy (%)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "72e029f5-6ca4-4d46-bfba-e9cc7fa95134",
   "metadata": {},
   "outputs": [],
   "source": [
    "X, Y = np.meshgrid(arr_framerate, arr_resol * 1000000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "34cdd77e-81aa-418f-b2e9-1a8cbe7eca7f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9.0, 30.0)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig4, ax4 = plt.subplots()\n",
    "khuj = ax4.contour(X, Y, arr_discrep_average)\n",
    "ax4.set_ylabel(r\"resolution ($\\mu$m/px)\")\n",
    "ax4.set_xlabel(r\"framerate (fps)\")\n",
    "ax4.clabel(khuj,\n",
    "          inline=True,       # draw labels on the contour lines\n",
    "          fmt='%1.0f',       # format string for the level numbers\n",
    "          fontsize=8)\n",
    "ax4.grid(True)\n",
    "ax4.set_ylim(9, 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "821be0ab-b024-4356-bd18-c83020582f94",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig4.savefig(\"contour_discrepancy.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "d513c9f7",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig4.savefig(\"contour_discrepancy.eps\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cdeaae37-8790-487a-acb0-1b121a2e20d8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
