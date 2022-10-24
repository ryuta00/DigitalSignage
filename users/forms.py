import datetime
from django import forms
from django.core.files.storage import default_storage


class ImageUploadForm(forms.Form):
    file = forms.ImageField()

    def save(self):
        """ファイルを保存するメソッド"""
        now_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')  # ファイル名に現在時刻を付与するため取得
        upload_file = self.files['file']  # フォームからアップロードファイルを取得
        file_name = default_storage.save(now_date + "_" + upload_file.name, upload_file)  # ファイルを保存 戻り値は実際に保存したファイル名
        return "/" + file_name
