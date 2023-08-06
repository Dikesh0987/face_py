import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://realtimefacedetection-42014-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference('students')

data = {
    "123":
        {
            "name": "Dikesh .",
            "major": "Robotics & Physics",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "231":
        {
            "name": "Dikesh",
            "major": "Robotics & Physics",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "321":
        {
            "name": "Dikesh",
            "major": "Robotics & Physics",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "132":
        {
            "name": "Dikesh .",
            "major": "Robotics & Physics",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
}

for key, value in data.items():
    ref.child(key).set(value)